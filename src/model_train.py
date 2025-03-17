import pandas as pd
import pytorch_forecasting as pf
from pytorch_forecasting.data import TimeSeriesDataSet
import pytorch_forecasting.models.temporal_fusion_transformer as tft

import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler

class ModelPrepData:
    """
    Description:
        prepares data for model training.

    Attributes:
        df (dataframe): DataFrame containing the data
        features (list): List of feature column names
        targets (list): List of column names for targets
        test_size (float): Test size to use for train/test split

    Methods:
        create_train_test(self)
    """

    def __init__(self, df, date_col, ticker_col, features, targets, test_start_date):
        self.df = df.copy()
        self.date_col = date_col
        
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.train_df = self.df[self.df[self.date_col] < test_start_date]
        self.test_df = self.df[self.df[self.date_col] >= test_start_date]
        self.train_df[self.date_col] = self.train_df[self.date_col].map(pd.Timestamp.toordinal)
        self.test_df[self.date_col] = self.test_df[self.date_col].map(pd.Timestamp.toordinal)
        
        self.ticker_col = ticker_col
        self.features = features
        self.targets = targets

    def prepare(self):
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
            scalers = {}
        )

        validation = TimeSeriesDataSet.from_dataset(training, self.df, predict=True, stop_randomization=True)

        test = TimeSeriesDataSet(
            self.test_df,
            time_idx=self.date_col,
            target=self.targets,
            group_ids=[self.ticker_col],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            time_varying_known_reals=self.features,
            time_varying_unknown_reals=self.targets,
            allow_missing_timesteps=True,
            scalers = {}
        )

        train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
        test_dataloader = test.to_dataloader(train=False, batch_size=64, num_workers=0)

        return train_dataloader, val_dataloader, test_dataloader
    
class CustomTFT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_regression, output_size_classification):
        super(CustomTFT, self).__init__()
        
        # Temporal Fusion Transformer (TFT)
        self.tft = tft.TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size, 
            dropout=0.1,
        )
        
        # Linear layer for regression (stock price prediction)
        self.regression_layer = nn.Linear(hidden_size, output_size_regression)
        
        # Linear layer for classification (stock movement direction)
        self.classification_layer = nn.Linear(hidden_size, output_size_classification)
        
    def forward(self, x):
        # Pass through the TFT model
        tft_output = self.tft(x)
        
        # Linear layer for regression
        regression_output = self.regression_layer(tft_output)
        
        # Linear layer for classification
        classification_output = self.classification_layer(tft_output)
        
        return regression_output, classification_output

class CreateModel:
    """
    Description:
        Class of methods to build my Model

    Attributes:
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data

    Methods:
        fit(self)
    """

    def __init__(self, train_dataloader, val_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def fit(self):
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the custom model
        input_size = len(self.train_dataloader.dataset.reals)
        hidden_size = 4096
        output_size_regression = 1 
        output_size_classification = 2 

        model = CustomTFT(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size_regression=output_size_regression,
            output_size_classification=output_size_classification
        ).to(device)

        # Define loss function and optimizer
        regression_criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Training loop
        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                x, y = batch
                x, y = x.to(device), {k: v.to(device) for k, v in y.items()}
                regression_output, classification_output = model(x)
                regression_loss = regression_criterion(regression_output, y['regression'])
                classification_loss = classification_criterion(classification_output, y['classification'])
                loss = regression_loss + classification_loss
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in self.val_dataloader:
                    x, y = batch
                    x, y = x.to(device), {k: v.to(device) for k, v in y.items()}
                    regression_output, classification_output = model(x)
                    regression_loss = regression_criterion(regression_output, y['regression'])
                    classification_loss = classification_criterion(classification_output, y['classification'])
                    loss = regression_loss + classification_loss
                    val_loss += loss.item()

                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss/len(self.val_dataloader)}')
            
            scheduler.step(val_loss)
            
        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                x, y = batch
                x, y = x.to(device), {k: v.to(device) for k, v in y.items()}
                regression_output, classification_output = model(x)
                regression_loss = regression_criterion(regression_output, y['regression'])
                classification_loss = classification_criterion(classification_output, y['classification'])
                loss = regression_loss + classification_loss
                test_loss += loss.item()

                print(f'Test Loss: {test_loss/len(self.test_dataloader)}')
                
        return model
