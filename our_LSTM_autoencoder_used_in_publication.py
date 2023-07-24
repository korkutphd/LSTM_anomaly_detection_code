# %% Import necessary libraries and dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import plotly.graph_objects as go
import pickle
import math

# Define dataset parameters
Dataset='bl_test10_47-63'
tstep='00008' # for KKK13 tstep=0.01, KKK14,15,16 tstep=0.008, KKK17 tstep=0.005

#%% Loading the data, preprocessing, and splitting into train/test sets

# Define the column names for the datasets
names=['energy','std','maxabs','skw','kurt','rms','clear','crest','shape','fe','p2p','spectrms','label'] #the columns names for LDV2
names_r=['energy_r','std_r','maxabs_r','skw_r','kurt_r','rms_r','clear_r','crest_r','shape_r','fe_r','p2p_r','spectrms_r','label']  #the columns names for LDV ratio

# Load the data and concatenate LDV2 with LDVRatio datasets
data_2= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names, sheet_name='ldv2')
data_2=data_2.drop(['label'],axis=1)
data_r= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names_r, sheet_name='ldv_r')
data=pd.concat([data_2,data_r],axis=1)
data_label=pd.DataFrame(data['label'])

# Drop columns based on prior correlation analysis
data_2=data_2.drop(['clear','p2p','maxabs','crest','spectrms'],axis=1)
data_r=data_r.drop(['kurt_r','skw_r','clear_r','shape_r','energy_r','std_r','p2p_r','crest_r','fe_r','rms_r'],axis=1)
data=pd.concat([data_2,data_r],axis=1)

# Apply log transform on skewed columns to make them more normally distributed
float_columns=[x for x in data.columns != 'label']
float_columns=data.columns[float_columns]
skew_columns=(data[float_columns].skew().sort_values(ascending=False))
skew_columns = skew_columns.loc[skew_columns>0.75]

for col in skew_columns.index.tolist():
    data[col]=np.log1p(data[col])

# Apply moving average for data smoothing
for col in float_columns:
    data[col]=data[col].rolling(window=250,center=False,min_periods=1).mean() # 350 for KKK13, 70 for KKK14, 15, 16 , 50 for KKK17

plt.plot(data['energy'])

# Splitting data into train, validation and test sets
start_index = 2500
end_index = 4000
x_train = data[start_index:end_index]
y_train = data['label'][start_index:end_index]
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train, y_train, test_size = 0.1, shuffle = False) 
x_test_1 = data[float_columns][:start_index]
y_test_1 = data['label'][:start_index]
x_test_2 = data[float_columns][end_index:]
y_test_2 = data['label'][end_index:]
x_testtest1 = np.concatenate([x_test_1, x_test_2], axis=0)
y_test1 = np.concatenate([y_test_1, y_test_2], axis=0)
x_testtest1=pd.DataFrame(x_testtest1,columns=['energy','std','skw','kurt','rms','shape','fe','maxabs_r','spectrms_r'])

# Standardize the data and save the mean and scale parameters for each feature
ss = StandardScaler()
ssmean=pd.DataFrame()
ssscale= pd.DataFrame()
ssmean.rename(columns=data.columns)
ssscale.rename(columns=data.columns)

for col in float_columns:
    x_train1[col] = ss.fit_transform(x_train1[[col]]).squeeze()
    ssmean[col]=ss.mean_
    ssscale[col]=ss.scale_
    x_val1[col] = ss.transform(x_val1[[col]]).squeeze()
    data[col]=ss.transform(data[[col]]).squeeze()

# Drop columns based on prior PCA analysis
pca_columns=pd.Series(['std','fe','energy','rms','kurt']) #for shape for 14,15,16
x_train1=x_train1[pca_columns]
x_val1=x_val1[pca_columns]
data=data[pca_columns]

# %% LSTM training

# Define the number of time steps for sequences
TIME_STEPS=20

# Function to create sequences from input data
def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for train, test, and entire dataset
x_train, y_train=[],[]
x_test, y_test=[],[]
x_train, y_train = create_sequences(x_train1, y_train1)
x_test, y_test = create_sequences(x_val1, y_val1)
data_feature, data_label =create_sequences(data,data_label)

# Print shapes of training and testing datasets
print(f'Training shape: {x_train.shape}')
print(f'Testing shape: {x_test.shape}')

# Define latent dimensions for the autoencoder
latent_dim = 2

# Autoencoder class definition
class AutoencoderModel:
    def __init__(self, X, latent_dim):
        # Encoder
        self.inputs = Input(shape=(X.shape[1], X.shape[2]))
        self.encoded = LSTM(128, activation='relu', return_sequences=True)(self.inputs)
        self.encoded = Dropout(rate=0.2)(self.encoded)
        self.encoded = LSTM(64, activation='relu', return_sequences=True)(self.encoded)
        self.encoded = Dropout(rate=0.2)(self.encoded)
        self.encoded = LSTM(16, activation='relu', return_sequences=True)(self.encoded)
        self.encoded = Dropout(rate=0.2)(self.encoded)
        self.encoded = LSTM(latent_dim)(self.encoded)
        # Decoder
        self.decoded = RepeatVector(X.shape[1])(self.encoded)
        self.decoded = LSTM(16, activation='relu', return_sequences=True)(self.decoded)
        self.decoded = Dropout(rate=0.2)(self.decoded)
        self.decoded = LSTM(64, activation='relu', return_sequences=True)(self.decoded)
        self.decoded = Dropout(rate=0.2)(self.decoded)
        self.decoded = LSTM(128, activation='relu', return_sequences=True)(self.decoded)
        self.output = TimeDistributed(Dense(X.shape[2]))(self.decoded)
        # Model
        self.model = Model(inputs=self.inputs, outputs=self.output)
        self.encoder = Model(self.inputs, self.encoded)
    
    # Method to get the autoencoder model
    def get_model(self):
        return self.model
    
    # Method to get the encoder
    def get_encoder(self):
        return self.encoder
    
# Create the autoencoder model and compile
autoencoder = AutoencoderModel(x_train, latent_dim)
model = autoencoder.get_model()
encoder = autoencoder.get_encoder()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
transfer_es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history=model.fit(x_train, x_train, epochs=50, batch_size=72, validation_split=0.1, callbacks= transfer_es)

#%% Post-Processing

# Plot training and validation loss
hfont = {'fontname':'Times New Roman'}
plt.grid(visible=None)
plt.ylim(ymin=0,ymax=round(max(max(history.history["val_loss"]),max(history.history["loss"])),1)*1.2)
plt.xlim(0,50)
plt.plot(history.history["val_loss"], label="Validation Loss",color='k',linestyle='-')
plt.plot(history.history["loss"], label="Training Loss",color='0.45',linestyle='--')
plt.xlabel("Number of Epoch",fontsize=24) #plt.xlabel("Number of Epoch", **hfont,fontsize=24)
plt.ylabel("Loss",fontsize=24)
plt.yticks(np.linspace(0,round(max(max(history.history["val_loss"]),max(history.history["loss"])),1)*1.2,num=5),fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16)
plt.show()

# Save the model
save_path = './SDS_'+Dataset+'.h5'
model.save(save_path)

# Calculate the train MAE loss and plot histogram
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
train_mae_loss = np.sum(train_mae_loss, axis=1)
train_mae_loss = train_mae_loss.reshape((-1))
train_mae_loss = train_mae_loss[50:-1,]
plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No. of samples")
plt.show()

# Compute and set the reconstruction loss threshold
threshold = np.max(train_mae_loss)*1+20*np.std(train_mae_loss) 
print("Reconstruction error threshold: ", threshold)

# Calculate the test MAE loss and plot histogram
x_test_pred = model.predict(data_feature)
test_mae_loss = np.mean(np.abs(x_test_pred - data_feature), axis=1)
test_mae_loss=np.sum(test_mae_loss, axis=1)
test_mae_loss = test_mae_loss.reshape((-1))
plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect anomalies based on the threshold
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# Identify which consecutive data points are anomalies
# A data point is considered anomalous if all samples from (i - timesteps + 1) to i are anomalies
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(data_feature) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

# Convert the anomaly detection results into a binary label format
predicted_labels = (test_mae_loss > threshold).astype(int)

# Load the original velocity data for visualization
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
fs=1250000
time=np.linspace(0, len(ldv2_acc_org)/fs, num=len(ldv2_acc_org))
time_mae=np.linspace(0,time[-1],num=len(test_mae_loss))

# Plot MAE loss over time
fig1, ax1 = plt.subplots()
ax1.plot(time_mae,test_mae_loss,c='k')
plt.xlabel("Time(s)",fontsize=16,fontname='Times New Roman')
plt.ylabel("MAE Loss",fontsize=16,fontname='Times New Roman')
plt.xlim(xmin=min(time), xmax=max(time))
fig1.set_figwidth(12)
fig1.set_figheight(4)
ax1.axhline(threshold, xmin=0.0, xmax=1.0, color='k',linestyle='--',linewidth=3)
timearray=np.append(np.arange(0, time[-1], step=1),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 

# Load and interpolate the downsampled velocity data to match the original data size
velocity_ldv2=pd.read_excel('LDV2_downsample_'+Dataset+'_tstep'+tstep+'.xls',header=None)
from scipy.interpolate import interp1d
def interpolate_b(a, b):
    # Create a function that interpolates b
    f = interp1d(np.arange(len(b)), b, kind='linear')
    # Create a new array that contains the interpolated values of b
    new_b = f(np.linspace(0, len(b)-1, len(a)))
    return new_b

# Example usage
velocity_ldv2 = pd.DataFrame(interpolate_b(np.array(data_label).squeeze() , np.array(velocity_ldv2).squeeze()))

# Plot velocity data with highlighted anomalies
fig1, ax1 = plt.subplots()
ax1.plot(velocity_ldv2)
ax1.plot(velocity_ldv2.iloc[anomalous_data_indices],color='r')
plt.xlabel("data points")
plt.ylabel("velocity(m/s)")

# Identify the exact indices of anomalies within the velocity data
anomalous_data_indices_fact = anomalous_data_indices.copy()
indx_fact=np.array(anomalous_data_indices_fact)
labels_pretrain=np.zeros(len(velocity_ldv2))
labels_pretrain[indx_fact.astype(int)]=1
  
# Loading original velocity data
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
ldv1_acc_org= pd.read_csv('LDV1_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
fs=1250000
print(ldv2_acc_org)

# Function to separate continuous anomaly indices into separate sublists
def separate_array(arr):
  sublists = []
  sublist = [arr[0]]
  for i in range(1, len(arr)):
    if arr[i] - arr[i-1] == 1:
      sublist.append(arr[i])
    else:
      sublists.append(sublist)
      sublist = [arr[i]]
  sublists.append(sublist)
  return sublists

indx_sep=separate_array(anomalous_data_indices)

# Calculate anomaly start and end indices in original data
anom_start_end=[]
anom_org=[]
for i in range(0,len(indx_sep)):
    indx_org_s=math.floor((indx_sep[i][0]/len(velocity_ldv2))*len(ldv2_acc_org))
    indx_org_e=math.floor((indx_sep[i][-1]/len(velocity_ldv2))*len(ldv2_acc_org))
    anom_start_end.append([indx_org_s,indx_org_e])
    anom_org.append(np.arange(indx_org_s,indx_org_e+1))

# Plotting Velocity vs Time
hfont = {'fontname':'Times New Roman'}
time=np.linspace(0, len(ldv2_acc_org)/fs, num=len(ldv2_acc_org))
fig2, ax2 = plt.subplots()
ax2.plot(time,ldv2_acc_org,color='k',linewidth=0.5)
ax2.plot(time,ldv1_acc_org,color='k',linestyle='--',linewidth=0.5)
for i in range(0,len(indx_sep)):
    ax2.plot(time[anom_org[i]],ldv2_acc_org.iloc[anom_org[i]],c='0.45',linewidth=0.5) #,marker="o",markersize=3,markerfacecolor='none'
    ax2.plot(time[anom_org[i]],ldv1_acc_org.iloc[anom_org[i]],c='0.45',linestyle='--',linewidth=0.5) #,marker="o",markersize=3,markerfacecolor='none'

plt.xlabel("Time(s)", **hfont,fontsize=16)
plt.ylabel("Velocity(m/s)", **hfont,fontsize=16)
plt.xlim(xmin=min(time), xmax=max(time))
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
fig2.set_figwidth(12)
fig2.set_figheight(4)

ldv2_acc_org = ldv2_acc_org.to_numpy()
lim_array=max(ldv2_acc_org)*1.5
lim=round(lim_array[0],2)
ax2.set_ylim([-1*lim,lim])
ax2.set_yticks(np.linspace(-1*lim,lim,num=5))
plt.yticks(fontsize=12)
ax2.grid(False)
ax2.legend(["LDV2","LDV1","LDV2_anomaly","LDV1_anomaly"],loc='upper right', fancybox=True, framealpha=0.1,ncol=4)

# Plotting time series features
feature="fe" #names=['energy','std','maxabs','kurt','rms','fe','p2p','spectrms','ssd','label']
time_d=np.linspace(0, len(ldv2_acc_org)/fs,num= len(data))
plt.plot(time_d,data[feature],'k')
plt.xlabel("Time(s)", **hfont)
plt.xlim(xmin=min(time), xmax=max(time))

lim_feat=round(abs(max(data[feature])),1)
ax2.set_yticks(np.linspace(-1*lim_feat,lim_feat,num=3))
plt.ylabel("Normalized Entropy", **hfont)

# Resizing predicted features to match the same dimension of the orginal time series
def extract_original_data(Xs, time_steps):
    X_original = pd.DataFrame(columns=[f'A{i}' for i in range(Xs.shape[2])])
    for i in range(0, Xs.shape[0], time_steps):
        X_original = X_original.append(pd.DataFrame(Xs[i], columns=[f'A{i}' for i in range(Xs.shape[2])]))
    X_original.reset_index(drop=True, inplace=True)
    return X_original

x_test_pred_orginalsize=[]
x_train_pred_orginalsize=[]
x_test_pred_orginalsize=extract_original_data(x_test,TIME_STEPS)
x_train_pred_orginalsize=extract_original_data(x_train,TIME_STEPS)
test_sizediff=len(x_test)-len(x_test_pred_orginalsize)
train_sizediff=len(x_train)-len(x_train_pred_orginalsize)
x_test_pred_orginalsize=pd.concat([x_test_pred_orginalsize,x_test_pred_orginalsize.tail(test_sizediff)],axis=0,ignore_index=True)
x_train_pred_orginalsize=pd.concat([x_train_pred_orginalsize,x_train_pred_orginalsize.tail(train_sizediff)],axis=0,ignore_index=True)

# Plotting in latent space (2D or 3D based on dimension)
plot_latent='True'
latent_option='y_test' # either anomalies or y_test or (data_feature or data_label)
if latent_option == 'data_label':
    latent_c=data_label
    latent_c=data_label.reshape(-1)
    encoderinput=data_feature
else:
    latent_c=y_test
    encoderinput=x_test
            
def plot_lat_predict(latent_dim,latent_c,encoderinput):      
    latent_representation = encoder.predict(encoderinput)
    latent_representation = np.array(latent_representation)
    colors = {True:'grey',False:'black'} 
    latent_c_color=np.vectorize(colors.get)(latent_c)

    if latent_dim == 2:
        plt.grid()
        ax = plt.axes
        plt.scatter(latent_representation[:, 0], latent_representation[:, 1],c=latent_c_color,alpha=0.8)
        plt.locator_params(axis='both', nbins=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
    else:
        ax = plt.axes(projection ="3d")
        plt.grid()
        ax.scatter3D(latent_representation[:, 0], latent_representation[:, 1], latent_representation[:, 2], c=latent_c_color,alpha=0.8)
        ax.view_init(0, 0)
        ax.xaxis._axinfo["grid"].update({"linewidth":0.2})
        ax.yaxis._axinfo["grid"].update({"linewidth":0.2})
        ax.zaxis._axinfo["grid"].update({"linewidth":0.2})
        ax.set_xticks([-max(abs(latent_representation[:, 0])), 0, max(abs(latent_representation[:, 0]))])
        ax.set_yticks([-max(abs(latent_representation[:, 1])), 0, max(abs(latent_representation[:, 1]))])
        ax.set_zticks([-max(abs(latent_representation[:, 2])), 0, max(abs(latent_representation[:, 2]))])
        ax.zaxis.set_tick_params(labelsize=8)
                
if plot_latent=='True':        
    plot_lat_predict(latent_dim,latent_c,encoderinput)

# Plotting Defect Annotations
# First set of defects
span=0.52
defects={"surface":[[8,9],[23,24],[31,32],[65,67],[76,77],[82,83],[96,97],[129,130],[159,160],[319,320]],"Joint":[[44,45],[49,50],[175,176],[204,205],[253,254],[341,342],[425,426],[492,493],[504,505]],"TD":[[153,155],[270,272],[313,316]]}
total_n=508
col=['g','r','y']
plt.axhline(y=0.0, color='k', linestyle='-',linewidth=10)
plt.xlim(xmax=total_n*span)
y=[0,0]

for index, defect in enumerate(defects):
    colr=col[index]
    for section in defects.get(defect):        
        firstspan=section[0]*span
        secondspan=section[1]*span
        defect_time=np.linspace(firstspan,secondspan,2)
        print(defect_time)
        plt.plot(defect_time,y,linestyle='-',linewidth=4,color=colr)

# Second set of defects
span=0.52
defects={"surface":[[8,9],[23,24],[30,31],[32,33],[58,59],[71,72],[77,78],[120,121],[153,154]],"Joint":[[36,37],[43,44],[48,49],[68,69],[114,115],[137,138],[162,163],[169,170],[186,187],[196,197],[247,248],[281,282]],"Bjoint":[[169,170],[196,197],[247,248]],"TD":[[148,149],[261,264],[306,309]]}
total_n=508
col=['g','r','b','y']
plt.axhline(y=0.0, color='k', linestyle='-',linewidth=10)
plt.xlim(xmax=total_n*span)
y=[0,0]

for index, defect in enumerate(defects):
    colr=col[index]
    for section in defects.get(defect):        
        firstspan=section[0]*span
        secondspan=section[1]*span
        defect_time=np.linspace(firstspan,secondspan,2)
        print(defect_time)
        plt.plot(defect_time,y,linestyle='-',linewidth=4,color=colr)
