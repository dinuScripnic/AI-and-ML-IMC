import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


def main():
    df = pd.read_csv("wines.csv")
    # print(df.head())
    # print(df.describe())
    # print(df.info())
    # we have 4 columns and 1599 rows
    # we have 3 columns with float data type and 1 column with int data type
    # we will use the density, chloorides and volatileacidity columns to predict the quality of the wine
    # we will use the quality column as the target variable
    # wine quality is in range from 3 to 8

    # first we will split the data into training and testing data
    training_data, test_data = train_test_split(df, test_size=0.2)
    x_train = training_data[["density", "chlorides", "volatileacidity"]]
    y_train = training_data["quality"]
    x_test = test_data[["density", "chlorides", "volatileacidity"]]
    y_test = test_data["quality"]
    # # plot the wine quality
    # plt.hist(df['quality'])
    # plt.show()
    # # plot the density
    # plt.hist(df['density'])
    # plt.show()
    # # plot the chloorides
    # plt.hist(df['chlorides'])
    # plt.show()
    # # plot the volatileacidity
    # plt.hist(df['volatileacidity'])
    # plt.show()

    # Create a linear regression model
    mlr = LinearRegression()  # create the Linear Regression model
    mlr.fit(x_train, y_train)  # fit the model to the training data
    y_pred = mlr.predict(x_train)  # predict the quality of the wine
    y_pred_test = mlr.predict(
        x_test
    )  # predict the quality of the wine for the testing data
    MAE_train = metrics.mean_absolute_error(
        y_train, y_pred
    )  # calculate the mean absolute error
    MAPE_train = metrics.mean_absolute_percentage_error(
        y_train, y_pred
    )  # calculate the mean absolute percentage error
    RMSE_train = metrics.mean_squared_error(
        y_train, y_pred, squared=False
    )  # calculate the root mean squared error
    R2_train = metrics.r2_score(y_train, y_pred)  # calculate the R2 score
    MAE_test = metrics.mean_absolute_error(
        y_test, y_pred_test
    )  # calculate the mean absolute error for the testing data
    MAPE_test = metrics.mean_absolute_percentage_error(
        y_test, y_pred_test
    )  # calculate the mean absolute percentage error for the testing data
    RMSE_test = metrics.mean_squared_error(
        y_test, y_pred_test, squared=False
    )  # calculate the root mean squared error for the testing data
    R2_test = metrics.r2_score(
        y_test, y_pred_test
    )  # calculate the R2 score for the testing data

    # save the results in a dataframe
    MLR_results = pd.DataFrame(
        {
            "Training Data": [MAE_train, MAPE_train, RMSE_train, R2_train],
            "Testing Data": [MAE_test, MAPE_test, RMSE_test, R2_test],
        },
        index=[
            "Mean Absolute Error",
            "Mean Absolute Percentage Error",
            "Root Mean Squared Error",
            "R2",
        ],
    )

    # Fitting
    # poly_li = list()
    MAE_li_tren = list()
    MAE_li_test = list()
    budget = 7
    for i in range(budget):
        poly = PolynomialFeatures(degree=i)
        x_poly_train = poly.fit_transform(x_train)
        x_poly_test = poly.transform(x_test)
        model = LinearRegression()
        model.fit(x_poly_train, y_train)
        y_poly_train = model.predict(x_poly_train)
        y_poly_test = model.predict(x_poly_test)
        MAE_train = metrics.mean_absolute_error(y_train, y_poly_train)
        MAE_test = metrics.mean_absolute_error(y_test, y_poly_test)
        MAE_li_tren.append(MAE_train)
        MAE_li_test.append(MAE_test)

    plt.scatter(x=range(budget), y=MAE_li_tren, linewidths=2, color="blue")
    plt.plot(range(budget), MAE_li_tren, color="blue")
    plt.scatter(x=range(budget), y=MAE_li_test, color="red")
    plt.plot(range(budget), MAE_li_test, color="red")
    plt.show()

    # in most of the cases the model starts to overfit at the degree of 4, so we will use the degree of 4
    # but rarely it improves the model at the degree of 5

    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(x_train)
    x_poly_test = poly.transform(x_test)
    poly_model = LinearRegression().fit(x_poly, y_train)
    y_pred = poly_model.predict(x_poly)
    y_pred_test = poly_model.predict(x_poly_test)
    MAE_train = metrics.mean_absolute_error(y_train, y_pred)
    MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)
    MAPE_train = metrics.mean_absolute_percentage_error(y_train, y_pred)
    MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_pred_test)
    RMSE_train = metrics.mean_squared_error(y_train, y_pred, squared=False)
    RMSE_test = metrics.mean_squared_error(y_test, y_pred_test, squared=False)
    R2_train = metrics.r2_score(y_train, y_pred)
    R2_test = metrics.r2_score(y_test, y_pred_test)

    R2_test = metrics.r2_score(y_test, poly_model.predict(poly.fit_transform(x_test)))
    PR_results = pd.DataFrame(
        {
            "Training Data": [MAE_train, MAPE_train, RMSE_train, R2_train],
            "Testing Data": [MAE_test, MAPE_test, RMSE_test, R2_test],
        },
        index=[
            "Mean Absolute Error",
            "Mean Absolute Percentage Error",
            "Root Mean Squared Error",
            "R2",
        ],
    )
    
    print(MLR_results)
    print(PR_results)


if __name__ == "__main__":
    main()
