from data_processing import get_df
from data_processing import create_graph_close
from data_processing import modify_df
from data_processing import create_graph_delta

from single_lstm import get_df_singular
from single_lstm import modify_df_singular
from single_lstm import single_lstm_model
from single_lstm import predict_singular

from datetime import datetime
import matplotlib.pyplot as plt

def singular_model():
    df = get_df_singular()
    x_train, y_train, x_test, y_test, scalar = modify_df_singular(df, 0.8, 60) # training_dataset_percentage = 0.8, x_train_len = 60
    model = single_lstm_model(x_train, y_train)
    predictions, rmse = predict_singular(model, x_test, y_test, scalar)

    # Plot the data
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.savefig('Actual vs Predicted Data.png')
    plt.xlabel('Date from 2020-10-30 (days)')
    plt.ylabel('Close Price ($)')

    plt.savefig('Actual vs Predicted Data.png') # Save graph as png file
    print(f"rmse:{rmse}")

    model.save('my_model.keras') # save the model to a file


def data_analysis(): 
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    company_name = ['APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON']
    end = datetime(2023, 10, 30)
    start = datetime(2020, 10, 30)

    df, company_list = get_df(stock_list, company_name, start, end)

    create_graph_close(company_name, company_list)

    create_graph_delta(company_name, company_list)

    
def multi_model():
    data_analysis()

def main():
    singular_model()
    #multi_model()

if __name__ == '__main__':
    main()