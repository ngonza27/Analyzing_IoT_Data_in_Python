#Course: https://www.datacamp.com/courses/analyzing-iot-data-in-python
#Remeber to be on the same directory where you are calling the python file!

# Imports
import requests
import pandas as pd

# Download data from URL
res = requests.get("https://assets.datacamp.com/production/repositories/4296/datasets/8f6b478697a8d05e10b7d535af67154549a4f38f/environ_MS83200MS_airtemp_600_30r.json")

# Convert the result
data_temp = res.json()
#print(data_temp)

# Convert json data to Dataframe
df_temp = pd.DataFrame(data_temp)

print(df_temp.head())

#------------------------------------------------------------------------------------------------------------------------------------------

# Misma funcionalidad pero ya solo con pandas

# Import pandas
import pandas as pd

# Load URL to Dataframe
df_temp = pd.read_json(URL)

# Print first 5 rows
print(df_temp.head(5))

# Print datatypes
print(df_temp.dtypes)

#------------------------------------------------------------------------------------------------------------------------------------------

#Understanding the data
import pandas as pd

# Read file from json
df_env = pd.read_json('./environmental.json')
# Print summary statistics

print(df_env.head())
print(df_env.info())
print(df_env.describe())

#------------------------------------------------------------------------------------------------------------------------------------------
#Retriving message using MQTT

# Import mqtt library
import paho.mqtt.subscribe as subscribe

# Retrieve one message
msg = subscribe.simple("datacamp/iot/simple", hostname="mqtt.datacamp.com")

# Print topic and payload
print(f"{msg.topic}, {msg.payload}")

#------------------------------------------------------------------------------------------------------------------------------------------
#Save Datastream

import json
import pandas as pd

# Define function to call by callback method
def on_message(client, userdata, message):
    # Parse the message.payload
    data = json.loads(message.payload)
    store.append(data)

#MQTT_HOST =paho/test/iot_course

# Connect function to mqtt datastream
subscribe.callback(on_message, "paho/test/iot_course", hostname="paho/test/iot_course")

df = pd.DataFrame(store)
print(df.head())

# Store DataFrame to csv, skipping the index
df.to_csv("datastream.csv", index=0)

#------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

#PLOTTING THE INFO
cols = ["temperature", "humidity", "pressure"]

# Create a line plot
df[cols].plot(title="Environmental data",
              secondary_y="pressure")

# Label X-Axis
plt.xlabel("Time")

# Show plot
plt.show()


#Histogram:
cols = ["temperature", "humidity", "pressure", "radiation"]

# Create a histogram
df[cols].hist(bins=30)

# Label Y-Axis
plt.ylabel("Frequency")

# Show plot
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------

#Missing data
import pandas as pd
# Print head of the DataFrame
print(data.head())

# Drop missing rows
data_clean = data.dropna()
print(data_clean.head())

# Forward-fill missing values
data_clean = data.fillna(method="ffill")
print(data_clean.head())

#----------------------------

# Calculate and print NA count
# Check missing values on time:
print(data.isna().sum())


# Searching for periods of time
# Resample data
data_res = data.resample("10min").last()

# Calculate and print NA count
print(data_res.isna().sum())

# Plot the dataframe
data_res.plot()

plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------

# Cache Datastream
cache = []

def on_message(client, userdata, message):
 	# Combine timestamp and payload
    data = f"{message.timestamp},{message.payload}"
    # Append data to cache
    cache.append(data)
    # Check cache length
    if len(cache) > MAX_CACHE:
        with Path("energy.txt").open("a") as f:
            # Save to file
            f.writelines(cache)
        # reset cache
        cache.clear()

# Connect function to mqtt datastream
subscribe.callback(on_message, topics="datacamp/energy", hostname=MQTT_HOST)

#Convert timestamp
# Convert the timestamp
df["ts"] = pd.to_datetime(df["ts"], unit="ms")

# Print datatypes and first observations
print(df.dtypes)
print(df.head())

#Prepare and visulaize incremental data

#Reformat data
# Replace the timestamp with the parsed timestamp 
df['ts'] = pd.to_datetime(df["ts"], unit="ms")
print(df.head())

# Pivot the DataFrame
df2 = pd.pivot_table(df, columns="device", values="val", index="ts")
print(df2.head())

# Resample DataFrame to 1min
df3 = df2.resample("1min").max().dropna()
print(df3.head())

df3.to_csv(TARGET_FILE)

#Analyzing energy counter data
# Resample df to 30 minutes
df_res = df.resample('30min').max()

# Get difference between values
df_diff = df_res.diff()

# Get the percent changed
df_pct = df_diff.pct_change()

# Plot the DataFrame
df_pct.plot()
plt.show()


#Concatenate dataframes
# Rename the columns
temperature.columns = ["temperature"]
humidity.columns =["humidity"]
windspeed.columns = ["windspeed"]

# Create list of dataframes
df_list = [temperature,humidity,windspeed]

# Concatenate files
environment = pd.concat([temperature,humidity,windspeed], axis=1)

# Print dataframe
print(environment.head())


#Combine and resample data
# Combine the dataframes
environ_traffic = pd.concat([environ, traffic], axis=1)

# Print first 5 rows
print(environ_traffic.head())

# Create agg logic
agg_dict = {"temperature": "max", "humidity": "max", "sunshine": "sum", 
            "light_veh": "sum", "heavy_veh": "sum",
            }

# Resample the dataframe 
environ_traffic_resampled = environ_traffic.resample("1h").agg(agg_dict)
print(environ_traffic_resampled.head())

#Correlation

#Heat maps
# Calculate correlation
corr = data.corr()

# Print correlation
print(corr)

# Create a heatmap
sns.heatmap(corr, annot=True)

# Show plot
plt.show()

#Pairplot
# Import required modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a pairplot
sns.pairplot(data)

# Show plot
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------
# Srandard deviation
# Calculate mean
data["mean"] = data["temperature"].mean()

# Calculate upper and lower limits
data["upper_limit"] = data["mean"] + (data["temperature"].std() * 3)
data["lower_limit"] = data["mean"] - (data["temperature"].std() * 3)

# Plot the dataframe
data.plot()

plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------

#Autocorrelation
# Plot traffic dataset
traffic[:"2018-11-10"].plot() #data traffic before 2018-11-10

# Show plot
plt.show()

# Import tsaplots
from statsmodels.graphics import tsaplots

# Plot autocorrelation
tsaplots.plot_acf("vehicles", lags=50)

# Show the plot
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------

#Seasonal decomposition
# Import modules
import statsmodels.api as sm

# Perform decompositon 
res = sm.tsa.seasonal_decompose(traffic["vehicles"])

# Print the seasonal component
print(res.seasonal)

# Plot the result
res.plot()

# Show the plot
plt.show()



#Seasonal Decomposition II
# Resample dataframe to 1h
df_seas = df.resample('1h').max()

# Run seasonal decompose
decomp = sm.tsa.seasonal_decompose(df_seas)

# Plot the timeseries
plt.title("Temperature")
plt.plot(df_seas["temperature"], label="temperature")

# Plot trend and seasonality
plt.plot(decomp.trend["temperature"], label="trend")
plt.plot(decomp.seasonal["temperature"], label="seasonal")
plt.legend()
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------

#TRAIN/TEST split
# Define the split day
limit_day = "2018-10-27"

# Split the data
train_env = environment[:limit_day]
test_env = environment[limit_day:]

# Print start and end dates
print(show_start_end(train_env))
print(show_start_end(test_env))

# Split the data into X and y
X_train = train_env.drop("target", axis=1)
y_train = train_env["target"]
X_test = test_env.drop("target", axis=1)
y_test = test_env["target"]

#------------------------------------------------------------------------------------------------------------------------------------------

#Logistic regresion

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Initialize the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict classes
print(logreg.predict(X_test))


#------------------------------------------------------------------------------------------------------------------------------------------

#Model performance

# Create LogisticRegression model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Score the model
print(logreg.score(X_train, y_train))
print(logreg.score(X_test, y_test))

#------------------------------------------------------------------------------------------------------------------------------------------

# Scaling data
# Scaling data helps the algorithm converge faster. 
# It also avoids having one feature dominate all other features.

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
sc = StandardScaler()

# Fit the scaler
sc.fit(environment)

# Print mean and variance
print(sc.mean_)
print(sc.var_)

#Scaling II
# Initialize StandardScaler
sc = StandardScaler()

# Fit the scaler
sc.fit(environment)

# Transform the data
environ_scaled = sc.transform(environment)

# Convert scaled data to DataFrame
environ_scaled = pd.DataFrame(environ_scaled, 
                              columns=environment.columns, 
                              index=environment.index)
print(environ_scaled.head())
plot_unscaled_scaled(environment, environ_scaled)

#------------------------------------------------------------------------------------------------------------------------------------------

#CREATING PIPELINE

# Import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create Scaler and Regression objects
sc = StandardScaler()
logreg = LogisticRegression()

# Create Pipeline
pl = Pipeline([
        ("scale", sc),
        ("logreg", logreg)
    ])

# Fit the pipeline and print predictions
pl.fit(X_train, y_train)
print(pl.predict(X_test))


#STORE PIPELINE

# Create Pipeline
pl = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression())
    ])

# Fit the pipeline
pl.fit(X_train, y_train)

# Store the model
with Path("pipeline.pkl").open('bw') as f:
	pickle.dump(pl, f)
  
# Load the pipeline
with Path("pipeline.pkl").open('br') as f:
	pl_loaded = pickle.load(f)

print(pl_loaded)

#------------------------------------------------------------------------------------------------------------------------------------------

#MODEL PREDICTIONS

# Create Pipeline
pl = Pipeline([
        ("scale",StandardScaler()),
  		 ("logreg", LogisticRegression())
        ])

# Fit the pipeline
pl.fit(X_train, y_train)

# Predict classes
predictions = pl.predict(X_test)

# Print results
print(predictions)

#------------------------------------------------------------------------------------------------------------------------------------------

#APPLY MODEL TO DATASTREAM

def model_subscribe(client, userdata, message):
    data = json.loads(message.payload)
    # Parse to DataFrame
    df = pd.DataFrame.from_records([data], index="timestamp", columns=cols)
    # Predict result
    category = pl.predict(df)
    if category[0] < 1:
        # Call business logic
        close_window(df,category)
    else:
        print("Nice Weather, nothing to do.")  

# Subscribe model_subscribe to MQTT Topic
subscribe.callback(model_subscribe, topic, hostname=MQTT_HOST)

#------------------------------------------------------------------------------------------------------------------------------------------
