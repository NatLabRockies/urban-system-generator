"""
Urban System Generator - Building Attribute Inference

This module provides the main inference functionality for predicting missing
building attributes using a trained neural network model.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from .model import ScaledInputMaskedNN


class USGInference:
    """
    Main class for building attribute inference using trained models.
    """
    
    def __init__(
        self,
        model_path: str,
        cat_scaler_path: str,
        num_scaler_path: str,
        encoding_dict_path: str,
        all_model_attributes: list,
    ):
        """
        Initialize the inference engine with model and preprocessing artifacts.
        
        Args:
            model_path: Path to the trained Keras model (.keras)
            cat_scaler_path: Path to categorical feature scaler (.pkl)
            num_scaler_path: Path to numerical feature scaler (.pkl)
            encoding_dict_path: Path to encoding dictionary (.json or .pkl)
            all_model_attributes: List of all feature names the model expects
        """
        self.model_path = Path(model_path)
        self.cat_scaler_path = Path(cat_scaler_path)
        self.num_scaler_path = Path(num_scaler_path)
        self.encoding_dict_path = Path(encoding_dict_path)
        self.all_model_attributes = all_model_attributes
        
        # Load model and scalers
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scalers, and encoding dictionary."""
        # Load model
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"ScaledInputMaskedNN": ScaledInputMaskedNN},
            compile=False,
        )
        
        # Load scalers
        with open(self.cat_scaler_path, "rb") as f:
            self.cat_scaler = pickle.load(f)
        
        with open(self.num_scaler_path, "rb") as f:
            self.num_scaler = pickle.load(f)
        
        # Load encoding dictionary (smart load for pickle or JSON)
        with open(self.encoding_dict_path, "rb") as f:
            first = f.peek(1) if hasattr(f, "peek") else f.read(1)
            f.seek(0)
            try:
                self.enc_dict = (
                    pickle.load(f)
                    if first not in (b"{", b"[")
                    else json.load(f)
                )
            except Exception:
                f.seek(0)
                self.enc_dict = json.load(f)
        
        # Extract column names
        self.cat_cols = list(self.enc_dict.keys())
        self.num_cols = list(self.num_scaler.feature_names_in_)
    
    def predict_missing_single(self, known_attrs: dict) -> dict:
        """
        Predict missing attributes for a single building.
        
        Args:
            known_attrs: Dictionary of known building attributes
                Example: {"Vintage": "1990s", "Geometry Floor Area": "1000-1499"}
        
        Returns:
            Complete dictionary of building attributes (known + predicted)
        """
        # Build one-row DataFrame
        df = pd.DataFrame([{c: np.nan for c in self.all_model_attributes}])
        for k, v in known_attrs.items():
            if k in df.columns:
                df.at[0, k] = v
        
        # Encode and scale
        df_enc = df.copy()
        for col in self.cat_cols:
            entry = self.enc_dict[col]
            fmap = entry["forward"] if isinstance(entry, dict) and "forward" in entry else entry
            df_enc[col] = df_enc[col].map(lambda x: fmap.get(x, fmap.get("nan", 0)))
        
        df_enc[self.cat_cols] = self.cat_scaler.transform(df_enc[self.cat_cols])
        df_enc[self.num_cols] = self.num_scaler.transform(df_enc[self.num_cols])
        
        # Predict missing values
        X = df_enc.fillna(0).astype(np.float32).values
        M = df.notna().astype(np.float32).values  # 1 = known, 0 = missing
        preds = self.model.predict((X, M), verbose=0)[0]
        
        # Fill only missing values
        for j, col in enumerate(self.all_model_attributes):
            if M[0, j] == 0:
                df_enc.iat[0, j] = preds[j]
        
        # Inverse transforms
        df_enc[self.num_cols] = self.num_scaler.inverse_transform(df_enc[self.num_cols])
        df_enc[self.cat_cols] = self.cat_scaler.inverse_transform(df_enc[self.cat_cols])
        
        # Decode categorical values
        for col in self.cat_cols:
            fmap_inv = {
                v: k for k, v in (
                    self.enc_dict[col]["forward"]
                    if isinstance(self.enc_dict[col], dict) and "forward" in self.enc_dict[col]
                    else self.enc_dict[col]
                ).items()
            }
            code = int(round(df_enc.at[0, col]))
            if code not in fmap_inv:
                # Clamp to nearest valid code
                code = min(fmap_inv, key=lambda k: (abs(k - code), k))
            df_enc.at[0, col] = fmap_inv[code]
        
        return df_enc.iloc[0].to_dict()
    
    def process_buildings_batch(
        self,
        input_csv_path: str,
        output_csv_path: str,
        id_col: str = "Building",
    ) -> str:
        """
        Process multiple buildings from CSV, predicting missing attributes.
        
        Args:
            input_csv_path: Path to input CSV with incomplete building data
            output_csv_path: Path to save completed building data
            id_col: Column name for building IDs
        
        Returns:
            Status message with output path
        """
        # Read input CSV
        df_in = pd.read_csv(input_csv_path)
        
        if id_col not in df_in.columns:
            raise KeyError(f"ID column '{id_col}' not found in CSV.")
        
        completed_rows = []
        total_buildings = len(df_in)
        
        print(f"Processing {total_buildings} buildings...")
        
        for idx, row in df_in.iterrows():
            # Extract known attributes (non-null values)
            known_attrs = {
                col: row[col] 
                for col in self.all_model_attributes 
                if col in row and pd.notna(row[col])
            }
            
            # Predict missing attributes
            completed_attrs = self.predict_missing_single(known_attrs)
            
            # Add building ID back
            completed_attrs[id_col] = row[id_col]
            completed_rows.append(completed_attrs)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total_buildings} buildings")
        
        # Convert to DataFrame
        df_out = pd.DataFrame(completed_rows)
        
        # Ensure ID column is first
        cols = [id_col] + [c for c in self.all_model_attributes if c in df_out.columns]
        df_out = df_out[cols]
        
        # Save to CSV
        df_out.to_csv(output_csv_path, index=False)
        
        return f"Successfully processed {total_buildings} buildings. Output saved to {output_csv_path}"
    
    def process_for_simulation(
        self,
        input_csv_path: str,
        output_csv_path: str,
        id_col: str = "Building",
    ) -> str:
        """
        Prepare completed building data for simulation.
        
        This is essentially the same as process_buildings_batch but can be
        extended with additional simulation-specific processing.
        
        Args:
            input_csv_path: Path to input CSV
            output_csv_path: Path to save simulation-ready CSV
            id_col: Column name for building IDs
        
        Returns:
            Status message
        """
        return self.process_buildings_batch(input_csv_path, output_csv_path, id_col)