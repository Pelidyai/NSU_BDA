import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split

from support.constants import TARGET_NAME
from support.functions import load_x_prepared_train_data, load_y_train_norm_data, smape_loss

x_data = load_x_prepared_train_data()
x_data = x_data.drop(['id'], axis=1)
# x_data = x_data[[SALARY_FROM_KEY, NAME_DESC_PREDICTION_KEY]]
y_data = load_y_train_norm_data()[TARGET_NAME]

x = np.asarray(x_data).astype('float32')
y = np.asarray(y_data).astype('float32')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# initialize Pool
train_pool = Pool(x_train, y_train)

# specify the training parameters
model = CatBoostRegressor(iterations=100,
                          depth=16,
                          learning_rate=1,
                          loss_function="MAPE")
#train the model
model.fit(train_pool)

test_pool = Pool(x_test)
result = model.predict(test_pool)
print("SMAPE", np.asarray(smape_loss(y_test, np.asarray(result).astype("float32"))).astype('float32').mean())
model.save_model('../models/final_catboost/model.catboost')
# # make the prediction using the resulting model
# preds = model.predict(test_pool)
# print(preds)