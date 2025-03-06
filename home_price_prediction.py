import pandas as pd
import matplotlib
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


def plot_3d_pred_homeprice(x,y,z):
    """
        :param x: Area of the home
        :param y: Number of bedrooms
        :param z: The predicted value of the home
        :return: None - No return value
    """
    fig = plot.figure()
    ax = plot.axes(projection='3d')
    ax.scatter(x,y,z)
    ax.set_title(label='3D plot Home Price Prediction')
    ax.set_xlabel('Area',fontsize=10)
    ax.set_ylabel('BedRooms',fontsize=10)
    ax.set_zlabel('Price',fontsize=10)
    plot.savefig('3dplt_pred_output.png')

if __name__ == "__main__":

    #read the csv file into a dataframe
    df = pd.read_csv('./home.csv',names=['Area','BedRooms','Price'])

    #extract the relevant independent and dependent features
    inputx = df.iloc[:,0:2]
    inputy = df.iloc[0:,2]
    print(inputx)
    print(inputy)

    #Supervised learning
    # split the data into train and test
    input_train,input_test,output_train,output_test = train_test_split(inputx,inputy,test_size=1/4,random_state=0)

    #instantiate a linear regression model
    model = LinearRegression()

    # train the regression model
    model.fit(input_train,output_train)
    pred_output = model.predict(input_test)

    # print to console the score pf the model
    print(f"Auto calculated r2_score : ", model.score(input_test,output_test))
    print(f"Manually calculated r2_score : ", r2_score(output_test,pred_output))
    print(f"Mean Sqrd Error : ", root_mean_squared_error(output_test,pred_output))
    print(f"model params coeff: { model.coef_}, intercept: {model.intercept_}")

    # configure matplotlib for saving to an image file
    matplotlib.use('Agg')
    # create an image of the plot
    plot_3d_pred_homeprice(input_test['Area'],input_test['BedRooms'],pred_output)




