#%%
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

#%%
test_size = 0.3
# Generate normally distributed samples
numbers_gamma = np.linspace(0, 2, num=10+1)[1:]
numbers_beta = np.linspace(0, 2, num=10+1)[1:]

numbers_shift_x = np.linspace(0, 10, num=100)
numbers_shift_w = np.linspace(0, 10, num=100)

record_list = []

eps_x = np.random.normal(loc=0, scale=1, size=1000)
eps_y = np.random.normal(loc=0, scale=1, size=1000)
eps_w = np.random.normal(loc=0, scale=1, size=1000)

ood_eps_x = np.random.normal(loc=0, scale=1, size=int(test_size*1000))
ood_eps_y = np.random.normal(loc=0, scale=1, size=int(test_size*1000))
ood_eps_w = np.random.normal(loc=0, scale=1, size=int(test_size*1000))

for beta, gamma, shift_x, shift_w in tqdm(itertools.product(numbers_beta, numbers_gamma, numbers_shift_x,numbers_shift_w)):
    record = {}
    record["beta"] = beta
    record["gamma"] = gamma
    record["shift_x"] = shift_x
    record["shift_w"] = shift_w

    x = eps_x
    y = beta*x + eps_y
    w = gamma*y + eps_w

    ood_x = ood_eps_x+ shift_x
    ood_y = beta*ood_x + ood_eps_y
    ood_w = gamma*ood_y + ood_eps_w + shift_w

    ####################### only x  ####################### 
    target = y

    # Combine the variables into a single feature matrix
    features = x.reshape(-1, 1)

    # Split the data into training and testing sets
    features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=0)

    # Create and fit the model
    model = LinearRegression()
    model.fit(features_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(features_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error: ", mse)
    record["mse_causal"] = mse

    # Make predictions on the ood test data
    ood_features_test = ood_x.reshape(-1, 1)
    ood_y_pred = model.predict(ood_features_test)

    # Calculate the mean squared error
    ood_mse = mean_squared_error(ood_y, ood_y_pred)
    # print("Out-of-domain Mean Squared Error: ", ood_mse) 
    record["ood_mse_causal"] = ood_mse

    ################# true x coefficient  ################# 
    target = y

    # Combine the variables into a single feature matrix
    features = x.reshape(-1, 1)

    # Split the data into training and testing sets
    features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=0)

    # Make predictions on the test data
    y_pred = (beta*features_test).reshape((-1,))
     # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    record["mse_true_causal"] = mse

    # Make predictions on the ood test data
    ood_features_test = ood_x.reshape(-1, 1)
    ood_y_pred = (beta*ood_features_test).reshape((-1,))

    # Calculate the mean squared error
    ood_mse = mean_squared_error(ood_y, ood_y_pred)
    # print("Out-of-domain Mean Squared Error: ", ood_mse) 
    record["ood_mse_true_causal"] = ood_mse

    ####################### x and w ####################### 
    target = y

    # Combine the variables into a single feature matrix
    features = np.column_stack((x,w))

    # Split the data into training and testing sets
    features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

    # Create and fit the model
    model = LinearRegression()
    model.fit(features_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(features_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error: ", mse)
    record["mse_all"] = mse

    # Make predictions on the ood test data
    ood_features_test = np.column_stack((ood_x,ood_w))
    ood_y_pred = model.predict(ood_features_test)

    # Calculate the mean squared error
    ood_mse = mean_squared_error(ood_y, ood_y_pred)
    # print("Out-of-domain Mean Squared Error: ", ood_mse) 
    record["ood_mse_all"] = ood_mse

    record_list.append(record.copy())


    ####################### only w  ####################### 
    target = y

    # Combine the variables into a single feature matrix
    features = w.reshape(-1, 1)

    # Split the data into training and testing sets
    features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=0)

    # Create and fit the model
    model = LinearRegression()
    model.fit(features_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(features_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error: ", mse)
    record["mse_w"] = mse

    # Make predictions on the ood test data
    ood_features_test = ood_w.reshape(-1, 1)
    ood_y_pred = model.predict(ood_features_test)

    # Calculate the mean squared error
    ood_mse = mean_squared_error(ood_y, ood_y_pred)
    # print("Out-of-domain Mean Squared Error: ", ood_mse) 
    record["ood_mse_w"] = ood_mse

    record_list.append(record.copy())



record_dataframe = pd.DataFrame(record_list)

#%%
record_beta_1_gamma_1 = record_dataframe[(record_dataframe["beta"]==1)&
                                         (record_dataframe["gamma"]==1)&
                                         (record_dataframe["shift_x"]<2)&
                                         (record_dataframe["shift_w"]<2)]

#%%
plt.plot(record_beta_1_gamma_1["shift_x"], record_beta_1_gamma_1["ood_mse_true_causal"], label="mse of true causal", color="#d55e00")
plt.plot(record_beta_1_gamma_1["shift_x"], record_beta_1_gamma_1["ood_mse_causal"], label="mse of est. causal",color="#de8f05",linestyle="dotted")

random.seed(33)
sample = random.sample(list(record_beta_1_gamma_1["shift_w"].unique()),5)
sample.sort()
# colors
# colors = sns.color_palette("colorblind").as_hex()
# colors.remove("#d55e00")
# colors.remove("#de8f05")
# colors.remove("#ca9161")
for index, value in enumerate(sample):
    record_beta_1_gamma_1_shift_w = record_beta_1_gamma_1[record_beta_1_gamma_1["shift_w"]==value]
    plt.plot(record_beta_1_gamma_1_shift_w["shift_x"], record_beta_1_gamma_1_shift_w["ood_mse_all"], label=f"mse of all with shift {round(value,2)}", color="#0173b2",alpha=1-index*0.2, linestyle="dashed")
plt.legend()
plt.savefig("plots_paper/simulations_x_shift.pdf")
plt.show()

#%%
ax = plt.axes(projection ='3d')
# plot_df = record_beta_1_gamma_1[["shift_x", "shift_w", "ood_mse_true_causal"]]
plot1 = ax.plot_trisurf(record_beta_1_gamma_1["shift_x"], record_beta_1_gamma_1["shift_w"].values, record_beta_1_gamma_1["ood_mse_true_causal"].values, label="mse of true causal", color="#d55e00", alpha=0.7)
plot2 = ax.plot_trisurf(record_beta_1_gamma_1["shift_x"].values, record_beta_1_gamma_1["shift_w"].values, record_beta_1_gamma_1["ood_mse_causal"].values, label="mse of est. causal", color="#de8f05",alpha=0.7)
plot3 = ax.plot_trisurf(record_beta_1_gamma_1["shift_x"].values, record_beta_1_gamma_1["shift_w"].values, record_beta_1_gamma_1["ood_mse_all"].values, label="mse of all", color="#0173b2", alpha=0.7)
plot4 = ax.plot_trisurf(record_beta_1_gamma_1["shift_x"].values, record_beta_1_gamma_1["shift_w"].values, record_beta_1_gamma_1["ood_mse_w"].values, label="mse of est. anticausal", color="#000000", alpha=0.7)
ax.set_xlabel("shift_x")
ax.set_ylabel("shift_w")

plot1._edgecolors2d = "#d55e00"
plot1._facecolors2d = "#d55e00"

plot2._edgecolors2d = "#de8f05"
plot2._facecolors2d = "#de8f05"

plot3._edgecolors2d = "#0173b2"
plot3._facecolors2d = "#0173b2"

plot4._edgecolors2d = "#000000"
plot4._facecolors2d = "#000000"

ax.legend()
plt.savefig("plots_paper/simulations_x_w_shift.pdf")
plt.show()

#%%
# test_size = 0.3
# # Generate normally distributed samples
# numbers_gamma = [1]
# numbers_beta = [1]

# numbers_shift_x = [1] #np.linspace(0, 10, num=100)
# numbers_shift_w = [1] #np.linspace(0, 10, num=100)

# record_list = []

# sigma_x = 1
# sigma_y = 1
# sigma_w = 1

# eps_x = np.random.normal(loc=0, scale=sigma_x, size=10000)
# eps_y = np.random.normal(loc=0, scale=sigma_y, size=10000)
# eps_w = np.random.normal(loc=0, scale=sigma_w, size=10000)

# # ood_eps_x = np.random.normal(loc=0, scale=1, size=int(test_size*1000))
# # ood_eps_y = np.random.normal(loc=0, scale=1, size=int(test_size*1000))
# # ood_eps_w = np.random.normal(loc=0, scale=1, size=int(test_size*1000))

# for beta, gamma, shift_x, shift_w in tqdm(itertools.product(numbers_beta, numbers_gamma, numbers_shift_x,numbers_shift_w)):
#     record = {}
#     record["beta"] = beta
#     record["gamma"] = gamma
#     record["shift_x"] = shift_x
#     record["shift_w"] = shift_w

#     x = eps_x
#     y = beta*x + eps_y
#     w = gamma*y + eps_w

#     gamma*beta*x+gamma*eps_y + eps_w 
#     # ood_x = ood_eps_x+ shift_x
#     # ood_y = beta*ood_x + ood_eps_y
#     # ood_w = gamma*ood_y + ood_eps_w + shift_w

#     ####################### only x  ####################### 
#     target = y

#     # Combine the variables into a single feature matrix
#     features = x.reshape(-1, 1)

#     # Split the data into training and testing sets
#     features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=0)

#     # Create and fit the model
#     model = LinearRegression()
#     model.fit(features_train, y_train)

#     ####################### x and w ####################### 
#     target = y

#     # Combine the variables into a single feature matrix
#     features = np.column_stack((x,w))

#     # Split the data into training and testing sets
#     features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

#     # Create and fit the model
#     model2 = LinearRegression()
#     model2.fit(features_train, y_train)

#     # # Theoretical considerations: 
#     # exp_coef_1 = (np.dot(w,w)*np.dot(x,y) - np.dot(x,w)*np.dot(w,y)) / (np.dot(x,x)*np.dot(w,w) - np.dot(x,w)**2)
#     # print(exp_coef_1)
#     # exp_coef_1 = (np.dot(w,w)*np.dot(x,beta*x+eps_y) - np.dot(x,w)*np.dot(w,1/gamma*(w-eps_w))) / (np.dot(x,x)*np.dot(w,w) - np.dot(x,w)**2)
#     # print(exp_coef_1)

#     # # input variables definition & simplify products
#     # longscp = gamma**2*beta**2*np.dot(x,x) + 2*gamma**2*beta*np.dot(x,eps_y) + 2*gamma*beta*np.dot(x,eps_w) + gamma**2*np.dot(eps_y,eps_y) + 2*gamma*np.dot(eps_y,eps_w) + np.dot(eps_w,eps_w)
#     # shortscp = gamma*beta*np.dot(x,x) + gamma*np.dot(x,eps_y) + np.dot(x,eps_w)
#     # # (gamma**2*beta**2*var_x + gamma**2*var_epsy) = np.dot(gamma*beta*x+gamma*eps_y + eps_w,gamma*beta*x+gamma*eps_y)
#     # term = gamma**2*beta**2*np.dot(x,x) + 2*gamma**2*beta*np.dot(x,eps_y) + gamma*beta*np.dot(x,eps_w) + gamma**2*np.dot(eps_y,eps_y) + gamma*np.dot(eps_y,eps_w)

#     # exp_coef_1 = (longscp*np.dot(x,beta*x+eps_y) - shortscp*1/gamma*term) / (np.dot(x,x)*longscp - shortscp**2)
#     # print(exp_coef_1)

#     # # input variances and covariances
#     # cov_x_epsy = 0 #np.dot(x,eps_y)
#     # cov_x_epsw = 0 #np.dot(x,eps_w)
#     var_epsy = sigma_y**2 #np.dot(eps_y,eps_y)
#     # cov_epsy_epsw = 0 # np.dot(eps_y,eps_w)
#     var_epsw = sigma_w**2 #np.dot(eps_w,eps_w)
#     var_x = sigma_x**2 #np.dot(x,x)
#     # longscp = gamma**2*beta**2*var_x + 2*gamma**2*beta*cov_x_epsy + 2*gamma*beta*cov_x_epsw + gamma**2*var_epsy + 2*gamma*cov_epsy_epsw + var_epsw
#     # shortscp = gamma*beta*var_x + gamma*cov_x_epsy + cov_x_epsw
#     # # term = np.dot(gamma*beta*x+gamma*eps_y + eps_w,gamma*beta*x+gamma*eps_y)
#     # term = gamma**2*beta**2*var_x + 2*gamma**2*beta*cov_x_epsy + gamma*beta*cov_x_epsw + gamma**2*var_epsy + gamma*cov_epsy_epsw
        
#     # exp_coef_1 = (longscp*beta*var_x +longscp*cov_x_epsy - shortscp*1/gamma*term) / (var_x*longscp - shortscp**2)
#     # print(exp_coef_1)

#     # # simplify for expected coefficient
#     # longscp = gamma**2*beta**2*var_x + gamma**2*var_epsy + var_epsw
#     # shortscp = gamma*beta*var_x 
#     # # term = np.dot(gamma*beta*x+gamma*eps_y + eps_w,gamma*beta*x+gamma*eps_y)
#     # term = gamma**2*beta**2*var_x + gamma**2*var_epsy

#     # exp_coef_1 = (longscp*beta*var_x - shortscp*1/gamma*term) / (var_x*longscp - shortscp**2)

#     # # input terms again
#     # exp_coef_1 = ((gamma**2*beta**2*var_x + gamma**2*var_epsy + var_epsw)*beta*var_x - gamma*beta*var_x*1/gamma*(gamma**2*beta**2*var_x + gamma**2*var_epsy)) / (var_x*(gamma**2*beta**2*var_x + gamma**2*var_epsy + var_epsw) - (gamma*beta*var_x)**2)

#     # Simplify
#     exp_coef_0 = var_epsw*beta / (gamma**2*var_epsy + var_epsw)
#     exp_coef_1 = (beta-exp_coef_0)/(gamma*beta)

#     # It holds
#     beta_est = model2.coef_[0]+model2.coef_[1]*beta*gamma
#     ####################### only w  ####################### 
#     target = y

#     # Combine the variables into a single feature matrix
#     features = w.reshape(-1, 1)

#     # Split the data into training and testing sets
#     features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=0)

#     # Create and fit the model
#     model3 = LinearRegression()
#     model3.fit(features_train, y_train)


# # %%
# print(f"OLS with X: {model.coef_}")
# print(f"OLS with X and W: {model2.coef_}")
# print(f"Expected OLS with X and W: {[exp_coef_0, exp_coef_1]}")
# print(f"OLS with  W: {model3.coef_}")
# # %%
