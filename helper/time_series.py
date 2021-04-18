"""Time Series

Help functions used to process and forecast each time series.

Author: Lok Ngan (lokdoesdata)
"""

# Package needed
import pandas as pd
import numpy as np
from pmdarima.arima import AutoARIMA
from sklearn.metrics import mean_squared_error as mse
import warnings

# Vanilla Python
from contextlib import closing
import multiprocessing
import csv
from datetime import (date, datetime)
from dateutil.relativedelta import relativedelta
from pathlib import Path

OUTPUT_PATH = Path(__file__).parents[1].joinpath('output').absolute()
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore', category=UserWarning)


class TimeSeries:
    """Simple class to handle time series forecasting.

    Parameters
    ----------
    data: DataFrame
        input data from Zillow DataFrame
    forecast_start: date or string
        the start of the forecasting period
    forecast_period: int, optional (defaults = 12)
        the number of periods to forecast
    training_period: int, optional (defaults = 36)
        the number of periods used to train the model
    """

    def __init__(
        self,
        data,
        forecast_start,
        forecast_period=12,
        training_period=36
    ):
        self.__data = data
        self.__forecast_date = self.__make_date_date(forecast_start)
        self.__forecast_str_date = self.__make_date_string(forecast_start)
        self.__training_period = training_period
        self.__forecast_period = forecast_period
        self.__start_train = self.__forecast_date - relativedelta(
            months=self.__training_period)
        self.__end_train = self.__forecast_date - relativedelta(months=1)
        self.__end_test = self.__forecast_date + relativedelta(
            months=self.__forecast_period-1)

    def forecast_by_state(self, state, threads=-1):
        output_file = OUTPUT_PATH.joinpath(
            f'{state}_Forecast.csv')

        if not output_file.is_file():
            self.__create_blank_file(output_file)

        completed_zip_codes = pd.read_csv(
            output_file,
            usecols=['zip_code'],
            dtype={'zip_code': 'str'},
            squeeze=True).tolist()

        cols = ['State', 'ZIP Code']

        cols.extend(pd.date_range(
            start=self.__start_train,
            end=self.__end_test,
            freq='M',
        ).astype(str).tolist())

        data = self.__data[
            (self.__data['State'] == state) &
            (~self.__data['ZIP Code'].isin(completed_zip_codes)) &
            (~self.__data[self.__make_date_string(self.__start_train)].isna()) &  # noqa
            (~self.__data[self.__make_date_string(self.__end_test)].isna())
        ][cols].copy()

        if threads == -1:
            num_cores = multiprocessing.cpu_count()
        else:
            num_cores = threads

        if len(data):
            with closing(multiprocessing.pool.ThreadPool(num_cores)) as pool:
                joined_output = pool.imap_unordered(
                    self._get_forecast_output,
                    data.itertuples(name=None)
                )
                with open(output_file, 'a') as f:
                    writer = csv.writer(f)
                    for row in joined_output:
                        writer.writerow(row)

    def _get_forecast_output(self, y):
        model = AutoARIMA(m=12, random_state=718)
        output = []
        zip_code = y[2]
        train = list(y[3:3+self.__training_period])
        test = list(y[3+self.__training_period:])

        try:
            model.fit(train)

            train_pred, train_ci = model.predict_in_sample(
                return_conf_int=True)
            train_lci = train_ci[:, 0]
            train_uci = train_ci[:, 1]

            train_rmse = np.sqrt(mse(train, train_pred))

            test_pred, test_ci = model.predict(
                n_periods=self.__forecast_period,
                return_conf_int=True)

            test_lci = test_ci[:, 0]
            test_uci = test_ci[:, 1]
            test_rmse = np.sqrt(mse(test, test_pred))

            model_order = str(model.model_)
        except:  # noqa
            train_pred = np.empty(self.__training_period).tolist()
            train_lci = np.empty(self.__training_period).tolist()
            train_uci = np.empty(self.__training_period).tolist()
            train_rmse = np.nan

            test_pred = np.empty(self.__forecast_period).tolist()
            test_lci = np.empty(self.__forecast_period).tolist()
            test_uci = np.empty(self.__forecast_period).tolist()
            test_rmse = np.nan

            model_order = np.nan

        output.append(zip_code)
        output.append(model_order)
        output.append(train_rmse)
        output.append(test_rmse)
        output.extend(train_pred)
        output.extend(train_lci)
        output.extend(train_uci)
        output.extend(test_pred)
        output.extend(test_lci)
        output.extend(test_uci)

        return(output)

    def __create_blank_file(self, file_name):
        columns = [
            'zip_code', 'model_order',
            'train_rmse', 'test_rmse'
        ]

        train_range = pd.date_range(
            start=self.__start_train,
            end=self.__end_train,
            freq='M'
        ).astype(str).tolist()

        test_range = pd.date_range(
            start=self.__forecast_date,
            end=self.__end_test,
            freq='M'
        ).astype(str).tolist()

        columns.extend([f'train_pred_{d}' for d in train_range])
        columns.extend([f'train_lci_{d}' for d in train_range])
        columns.extend([f'train_uci_{d}' for d in train_range])
        columns.extend([f'test_pred_{d}' for d in test_range])
        columns.extend([f'test_lci_{d}' for d in test_range])
        columns.extend([f'test_uci_{d}' for d in test_range])

        df = pd.DataFrame(columns=columns)
        df.to_csv(file_name, index=False)

    def __make_date_date(self, input_date):
        if isinstance(input_date, str):
            output_date = datetime.strptime(input_date, '%m/%d/%Y').date()
        elif isinstance(input_date, date):
            output_date = input_date
        else:
            raise TypeError('input_date must be str or date')

        return(output_date)

    def __make_date_string(self, input_date):
        if isinstance(input_date, str):
            output_date = self.__make_date_date(input_date)
        elif isinstance(input_date, date):
            output_date = input_date
        else:
            raise TypeError('input_date must be str or date')

        return(output_date.strftime('%Y-%m-%d'))

    def __make_empty_list(self, length_of_list):
        return(np.empty(length_of_list).tolist())
