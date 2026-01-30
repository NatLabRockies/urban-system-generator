"""
Urban System Generator - Building Attribute Inference

This module provides the main inference functionality for predicting missing
building attributes using a trained neural network model.
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Configure TensorFlow BEFORE importing it to prevent hanging in subprocess environments
# This is critical when USG is called from other CLIs (e.g., URBANopt Ruby CLI)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress TF warnings
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # Disable oneDNN optimizations
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')  # Force CPU-only (avoid GPU issues)

import tensorflow as tf

# Configure TensorFlow threading to prevent hanging in subprocess environments
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras import layers
from .model import ScaledInputMaskedNN
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
        all_model_attributes: list = None,  # Made optional - will auto-detect if None
    ):
        """
        Initialize the inference engine with model and preprocessing artifacts.
        
        Args:
            model_path: Path to the trained Keras model (.keras)
            cat_scaler_path: Path to categorical feature scaler (.pkl)
            num_scaler_path: Path to numerical feature scaler (.pkl)
            encoding_dict_path: Path to encoding dictionary (.json or .pkl)
            all_model_attributes: List of all feature names (optional - will auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.cat_scaler_path = Path(cat_scaler_path)
        self.num_scaler_path = Path(num_scaler_path)
        self.encoding_dict_path = Path(encoding_dict_path)
        
        # Load model and scalers
        self._load_artifacts()
        
        # Auto-detect model attributes if not provided
        if all_model_attributes is None:
            self.all_model_attributes = self._detect_model_attributes()
        else:
            self.all_model_attributes = all_model_attributes
    
    def _detect_model_attributes(self):
        """
        Automatically detect model attributes from the scalers and encoding dictionary.
        Returns the correct ordered list of all model attributes.
        """
        # Categorical columns from encoding dictionary
        cat_cols = list(self.enc_dict.keys())
        
        # Numerical columns from scaler
        num_cols = list(self.num_scaler.feature_names_in_)
        
        # Combined list - categorical first, then numerical
        # This is the standard order for most models
        all_attributes = cat_cols + num_cols
        
        return all_attributes
    
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
        # Build one-row DataFrame with proper dtype handling
        df = pd.DataFrame([{c: np.nan for c in self.all_model_attributes}])
        
        # Set dtypes appropriately to avoid FutureWarnings
        for col in self.cat_cols:
            df[col] = df[col].astype('object')
        for col in self.num_cols:
            df[col] = df[col].astype('float64')
            
        # Fill in known attributes
        for k, v in known_attrs.items():
            if k in df.columns:
                df.at[0, k] = v
        
        # Encode and scale
        df_enc = df.copy()
        for col in self.cat_cols:
            entry = self.enc_dict[col]
            fmap = entry["forward"] if isinstance(entry, dict) and "forward" in entry else entry
            # Convert values to string for lookup since encoding dict has string keys
            # This handles cases where CSV reads numeric values (e.g., 3) but dict has string keys (e.g., "3")
            def encode_value(x, fmap=fmap):
                if pd.isna(x):
                    return fmap.get("nan", 0)
                # Try direct lookup first
                if x in fmap:
                    return fmap[x]
                # Try string conversion for numeric values
                str_x = str(x)
                if str_x in fmap:
                    return fmap[str_x]
                # Try removing trailing .0 for float-like strings (e.g., "3.0" -> "3")
                if isinstance(x, float) and x == int(x):
                    int_str = str(int(x))
                    if int_str in fmap:
                        return fmap[int_str]
                # Fallback to nan encoding
                return fmap.get("nan", 0)
            df_enc[col] = df_enc[col].map(encode_value)
        
        df_enc[self.cat_cols] = self.cat_scaler.transform(df_enc[self.cat_cols])
        df_enc[self.num_cols] = self.num_scaler.transform(df_enc[self.num_cols])
        
        # Predict missing values
        X = df_enc.fillna(0).astype(np.float32).values
        M = df.notna().astype(np.float32).values  # 1 = known, 0 = missing
        
        # Use direct model call instead of predict() - more reliable in subprocess environments
        # model.predict() can hang when called from external CLIs due to threading issues
        preds = self.model((X, M), training=False).numpy()[0]
        
        # Fill only missing values
        for j, col in enumerate(self.all_model_attributes):
            if M[0, j] == 0:
                df_enc.iat[0, j] = preds[j]
        
        # Inverse transforms
        df_enc[self.num_cols] = self.num_scaler.inverse_transform(df_enc[self.num_cols])
        df_enc[self.cat_cols] = self.cat_scaler.inverse_transform(df_enc[self.cat_cols])
        
        
        # Decode categorical values WITHOUT writing strings into df_enc
        result = df_enc.iloc[0].to_dict()
        decode_warnings = []

        for col in self.cat_cols:
            entry = self.enc_dict[col]
            fmap = entry["forward"] if isinstance(entry, dict) and "forward" in entry else entry
            fmap_inv = {v: k for k, v in fmap.items()}

            raw_val = result.get(col, np.nan)
            if pd.isna(raw_val):
                continue

            code = int(round(float(raw_val)))

            if code not in fmap_inv:
                nearest_code = min(fmap_inv, key=lambda k: (abs(k - code), k))
                msg = (
                    f"[USGInference] Column '{col}': predicted code {code} "
                    f"not valid, snapped to nearest {nearest_code} "
                    f"('{fmap_inv[nearest_code]}')"
                )
                decode_warnings.append(msg)
                logger.warning(msg)
                code = nearest_code

            result[col] = fmap_inv[code]

        if decode_warnings:
            result["_decode_warnings"] = decode_warnings

        return result
    
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