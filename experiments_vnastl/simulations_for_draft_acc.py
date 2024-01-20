#%%
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tqdm import tqdm
#%%
test_size = 0.3
# Generate normally distributed samples
numbers_gamma = [1] #np.linspace(0, 2, num=10+1)[1:]
numbers_beta = [1] #np.linspace(0, 2, num=10+1)[1:]

numbers_shift_x = np.linspace(-2, 2, num=100+1)
numbers_shift_w = np.linspace(-2, 2, num=100+1)

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
    y_class = (y>0)

    ood_x = ood_eps_x+ shift_x
    ood_y = beta*ood_x + ood_eps_y
    ood_w = gamma*ood_y + ood_eps_w + shift_w
    ood_y_class = (ood_y>0)

    target = y_class
    ####################### only x  ####################### 
    
    # Combine the variables into a single feature matrix
    features = x.reshape(-1, 1)

    # Split the data into training and testing sets
    features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=0)

    # Create and fit the model
    model = LogisticRegression(penalty='none')
    model.fit(features_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(features_test)

    # Calculate the mean squared error
    acc = accuracy_score(y_test, y_pred)
    # print("Mean Squared Error: ", acc)
    record["acc_causal"] = acc

    # Make predictions on the ood test data
    ood_features_test = ood_x.reshape(-1, 1)
    ood_y_pred = model.predict(ood_features_test)

    # Calculate the mean squared error
    ood_acc = accuracy_score(ood_y_class, ood_y_pred)
    # print("Out-of-domain Mean Squared Error: ", ood_acc) 
    record["ood_acc_causal"] = ood_acc

    ################# true x coefficient  ################# 

    # Combine the variables into a single feature matrix
    features = x.reshape(-1, 1)

    # Split the data into training and testing sets
    features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=0)

    # Make predictions on the test data
    y_pred = ((beta*features_test).reshape((-1,))>0)
     # Calculate the mean squared error
    acc = accuracy_score(y_test, y_pred)
    record["acc_true_causal"] = acc

    # Make predictions on the ood test data
    ood_features_test = ood_x.reshape(-1, 1)
    ood_y_pred = ((beta*ood_features_test).reshape((-1,))>0)

    # Calculate the mean squared error
    ood_acc = accuracy_score(ood_y_class, ood_y_pred)
    # print("Out-of-domain Mean Squared Error: ", ood_acc) 
    record["ood_acc_true_causal"] = ood_acc

    ####################### x and w ####################### 

    # Combine the variables into a single feature matrix
    features = np.column_stack((x,w))

    # Split the data into training and testing sets
    features_train, features_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

    # Create and fit the model
    model = LogisticRegression(penalty='none')
    model.fit(features_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(features_test)

    # Calculate the mean squared error
    acc = accuracy_score(y_test, y_pred)
    # print("Mean Squared Error: ", acc)
    record["acc_all"] = acc

    # Make predictions on the ood test data
    ood_features_test = np.column_stack((ood_x,ood_w))
    ood_y_pred = model.predict(ood_features_test)

    # Calculate the mean squared error
    ood_acc = accuracy_score(ood_y_class, ood_y_pred)
    # print("Out-of-domain Mean Squared Error: ", ood_acc) 
    record["ood_acc_all"] = ood_acc

    record_list.append(record.copy())

record_dataframe = pd.DataFrame(record_list)

#%%
record_beta_1_gamma_1 = record_dataframe[(record_dataframe["beta"]==1)&
                                         (record_dataframe["gamma"]==1)&
                                         (record_dataframe["shift_x"]<2)&
                                         (record_dataframe["shift_w"]<2)]

#%%
plt.plot(record_beta_1_gamma_1["shift_x"], record_beta_1_gamma_1["ood_acc_true_causal"], label="acc of true causal", color="#d55e00",linestyle="dotted")
plt.plot(record_beta_1_gamma_1["shift_x"], record_beta_1_gamma_1["ood_acc_causal"], label="acc of est. causal",color="#de8f05")

record_beta_1_gamma_1_shift_w = record_beta_1_gamma_1[record_beta_1_gamma_1["shift_w"]==0]
plt.plot(record_beta_1_gamma_1_shift_w["shift_x"], record_beta_1_gamma_1_shift_w["ood_acc_all"], label=f"acc of all", color="#0173b2",alpha=1-0*0.2)

random.seed(0)
sample = random.sample(list(record_beta_1_gamma_1["shift_w"].unique()),5)
sample.sort()
# colors
# colors = sns.color_palette("colorblind").as_hex()
# colors.remove("#d55e00")
# colors.remove("#de8f05")
# colors.remove("#ca9161")
for index, value in enumerate(sample):
    record_beta_1_gamma_1_shift_w = record_beta_1_gamma_1[record_beta_1_gamma_1["shift_w"]==value]
    plt.plot(record_beta_1_gamma_1_shift_w["shift_x"], record_beta_1_gamma_1_shift_w["ood_acc_all"], label=f"acc of all with shift {round(value,2)}", color="#0173b2",alpha=1-(index+1)*0.15, linestyle="dashed")
plt.xlabel("shift_x")
plt.ylabel("acc")
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig("plots_paper/simulations_acc_x_shift.pdf")
plt.show()

#%%
record_beta_1_gamma_1_shift_w = record_beta_1_gamma_1[record_beta_1_gamma_1["shift_x"]==0]
plt.plot(record_beta_1_gamma_1_shift_w["shift_w"], record_beta_1_gamma_1_shift_w["ood_acc_all"], label=f"acc of all", color="#0173b2",alpha=1-0*0.2)

record_beta_1_gamma_1_shift_x = record_beta_1_gamma_1[record_beta_1_gamma_1["shift_x"]==0]
plt.plot(record_beta_1_gamma_1_shift_x["shift_w"], record_beta_1_gamma_1_shift_x["ood_acc_true_causal"], label="acc of true causal", color="#d55e00")
plt.plot(record_beta_1_gamma_1_shift_x["shift_w"], record_beta_1_gamma_1_shift_x["ood_acc_causal"], label="acc of est. causal",color="#de8f05",linestyle="dotted")

random.seed(0)
sample = random.sample(list(record_beta_1_gamma_1["shift_x"].unique()),5)
sample.sort()
# colors
# colors = sns.color_palette("colorblind").as_hex()
# colors.remove("#d55e00")
# colors.remove("#de8f05")
# colors.remove("#ca9161")
for index, value in enumerate(sample):
    record_beta_1_gamma_1_shift_w = record_beta_1_gamma_1[record_beta_1_gamma_1["shift_x"]==value]
    plt.plot(record_beta_1_gamma_1_shift_w["shift_w"], record_beta_1_gamma_1_shift_w["ood_acc_causal"], label=f"acc of causal with shift {round(value,2)}", color="#de8f05",alpha=1-(index+1)*0.15, linestyle="dashed")

plt.xlabel("shift_w")
plt.ylabel("acc")
plt.tight_layout()
plt.legend()
plt.savefig("plots_paper/simulations_acc_w_shift.pdf")
plt.show()
#%%
ax = plt.axes(projection ='3d')
plot_df = record_beta_1_gamma_1[["shift_x", "shift_w", "ood_acc_true_causal"]]
plot1 = ax.plot_trisurf(record_beta_1_gamma_1["shift_x"], record_beta_1_gamma_1["shift_w"].values, record_beta_1_gamma_1["ood_acc_true_causal"].values, label="acc of true causal", color="#d55e00", alpha=0.7)
plot2 = ax.plot_trisurf(record_beta_1_gamma_1["shift_x"].values, record_beta_1_gamma_1["shift_w"].values, record_beta_1_gamma_1["ood_acc_causal"].values, label="acc of est. causal", color="#de8f05",alpha=0.7)
plot3 = ax.plot_trisurf(record_beta_1_gamma_1["shift_x"].values, record_beta_1_gamma_1["shift_w"].values, record_beta_1_gamma_1["ood_acc_all"].values, label="acc of all", color="#0173b2", alpha=0.7)
ax.set_xlabel("shift_x")
ax.set_ylabel("shift_w")
ax.set_zlabel("acc")

plot1._edgecolors2d = "#d55e00"
plot1._facecolors2d = "#d55e00"

plot2._edgecolors2d = "#de8f05"
plot2._facecolors2d = "#de8f05"

plot3._edgecolors2d = "#0173b2"
plot3._facecolors2d = "#0173b2"

ax.legend()
plt.tight_layout()
plt.savefig("plots_paper/simulations_acc_x_w_shift.pdf")
plt.show()

# %%
