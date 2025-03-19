import pandas as pd
import pytorch_forecasting as pf
from pytorch_forecasting.data import TimeSeriesDataSet
import pytorch_forecasting.models.temporal_fusion_transformer as tft

import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPrepData:
    """
    Description:
        Prepares data for model training.

    Attributes:
        df (dataframe): DataFrame containing the data
        date_col (str): Column name for dates
        ticker_col (str): Column name for ticker
        features (list): List of feature column names
        targets (list): List of column names for targets
        test_start_date (str): Start date for the test set

    Methods:
        prepare(self): Prepares the data and returns dataloaders for training, validation, and testing
    """

    def __init__(self, df, date_col, ticker_col, features, targets, test_start_date):
        self.df = df.copy()
        self.date_col = date_col
        
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        self.train_df = self.df[self.df[self.date_col] < test_start_date]
        val_start_date = pd.to_datetime(test_start_date) - pd.DateOffset(months=6)
        self.val_df = self.df[(self.df[self.date_col] >= val_start_date) & (self.df[self.date_col] < test_start_date)]
        self.test_df = self.df[self.df[self.date_col] >= test_start_date]

        # Print the size of each dataframe
        logging.info(f'Training DataFrame size: {self.train_df.shape}')
        logging.info(f'Validation DataFrame size: {self.val_df.shape}')
        logging.info(f'Test DataFrame size: {self.test_df.shape}')
        
        self.train_df[self.date_col] = self.train_df[self.date_col].map(pd.Timestamp.toordinal)
        self.val_df[self.date_col] = self.val_df[self.date_col].map(pd.Timestamp.toordinal)
        self.test_df[self.date_col] = self.test_df[self.date_col].map(pd.Timestamp.toordinal)
        
        self.ticker_col = ticker_col
        self.features = features
        self.targets = targets

    def prepare(self):
        """
        Prepares the data and returns dataloaders for training, validation, and testing.
        """
        self.df = self.df[self.features + [self.date_col] + self.targets]
        
        # Create TimeSeriesDataSet for pytorch-forecasting
        max_encoder_length = 24
        max_prediction_length = 10

        training = TimeSeriesDataSet(
            self.train_df,
            time_idx=self.date_col,
            target=self.targets,
            group_ids=[self.ticker_col],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            time_varying_known_reals=self.features,
            time_varying_unknown_reals=self.targets,
            allow_missing_timesteps=True,
            scalers={}
        )
        validation = TimeSeriesDataSet.from_dataset(training, self.val_df, predict=False, stop_randomization=True)
        test = TimeSeriesDataSet.from_dataset(training, self.test_df, predict=True, stop_randomization=True)

        train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
        test_dataloader = test.to_dataloader(train=False, batch_size=64, num_workers=0)

        return train_dataloader, val_dataloader, test_dataloader, self.train_df

# Custom Multi-Task Loss Function
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: weight for regression loss (1 - alpha used for classification loss)
        """
        super().__init__()
        self.alpha = alpha
        self.regression_loss = nn.MSELoss()  # Loss for percent change
        self.classification_loss = nn.BCEWithLogitsLoss()  # Loss for up/down direction

    def forward(self, y_pred, y_true):
        """
        y_pred: tuple (regression_output, classification_output)
        y_true: tuple (true_percent_change, true_direction)
        """
        percent_change_pred, direction_pred = y_pred
        percent_change_true, direction_true = y_true

        # Compute loss
        reg_loss = self.regression_loss(percent_change_pred, percent_change_true)
        class_loss = self.classification_loss(direction_pred, direction_true)

        # Weighted sum of losses
        return self.alpha * reg_loss + (1 - self.alpha) * class_loss

# 2. Custom TFT Model with Dual Output Heads
class MultiTaskTFT(tft.TemporalFusionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.output_layer[0].in_features  # Get last hidden size

        # Separate output heads for regression and classification
        self.regression_head = nn.Linear(hidden_size, 1)  # Predicts percent change
        self.classification_head = nn.Linear(hidden_size, 2)  # Predicts up/down

    def forward(self, x):
        out = super().forward(x)  # TFT base forward pass

        # Get last hidden representation
        hidden = out["prediction"]  # Shape: (batch, time, hidden_size)

        # Compute separate outputs
        percent_change_pred = self.regression_head(hidden).squeeze(-1)  # Continuous output
        direction_pred = self.classification_head(hidden).squeeze(-1)  # Logits for classification

        return percent_change_pred, direction_pred  # Return both outputs

class CreateModel:
    """
    Description:
        Class of methods to build and train the model.

    Attributes:
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        test_dataloader (DataLoader): DataLoader for test data
        patience (int): Number of epochs to wait for improvement before early stopping

    Methods:
        fit(self): Trains the model
    """

    def __init__(self, train_dataloader, val_dataloader, test_dataloader, train_df, regloss_weight=0.5, patience=10):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.patience = patience
        self.train_df = train_df

    def fit(self):
        """
        Trains the model using the provided dataloaders.
        """
        try:
            # Check if GPU is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize the custom model
            hidden_size = 4096  # Placeholder for hyperparameter tuning
            output_size_regression = 1 
            output_size_classification = 2

            model = MultiTaskTFT.from_dataset(
                self.train_df,
                learning_rate=0.001,
                loss=MultiTaskLoss(alpha=regloss_weight),  # Adjust alpha for regression vs classification weighting
                hidden_size=hidden_size,
                attention_head_size=4,
                dropout=0.3,
            ).to(device)

            # Define optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  # Placeholder for hyperparameter tuning
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            # Early stopping parameters
            best_val_loss = np.inf
            patience_counter = 0

            # Training loop
            num_epochs = 10  # Placeholder for hyperparameter tuning
            for epoch in range(num_epochs):
                model.train()
                for batch in self.train_dataloader:
                    optimizer.zero_grad()
                    x, y = batch
                    x, y = x.to(device), {k: v.to(device) for k, v in y.items()}
                    percent_change_pred, direction_pred = model(x)
                    loss = model.loss((percent_change_pred, direction_pred), (y['target_percent_gain'], y['target_direction']))
                    loss.backward()
                    optimizer.step()

                # Validation loop
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in self.val_dataloader:
                        x, y = batch
                        x, y = x.to(device), {k: v.to(device) for k, v in y.items()}
                        percent_change_pred, direction_pred = model(x)
                        loss = model.loss((percent_change_pred, direction_pred), (y['target_percent_gain'], y['target_direction']))
                        val_loss += loss.item()

                    val_loss /= len(self.val_dataloader)
                    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
                
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logging.info("Early stopping triggered")
                    model.load_state_dict(best_model)
                    break
            
            # Evaluate on test set
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in self.test_dataloader:
                    x, y = batch
                    x, y = x.to(device), {k: v.to(device) for k, v in y.items()}
                    percent_change_pred, direction_pred = model(x)
                    loss = model.loss((percent_change_pred, direction_pred), (y['target_percent_gain'], y['target_direction']))
                    test_loss += loss.item()

                logging.info(f'Test Loss: {test_loss/len(self.test_dataloader)}')
                    
            return model

        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise