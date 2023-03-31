from sklearn import model_selection
import sys
sys.path.append('../')
from ArtificialIntelligenceTest import network
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Build model
#####################
input_dim = 5
hidden_dim = 32
num_layers = 3
output_dim = 2
num_seeds = 4
look_back = 48  # choose sequence length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork():
    def __init__(self, market, data):
        self.market = market
        self.dataHistoDict = data
        self.models = []
        self.optimisers = []
        for i in range(num_seeds):
            self.models.append(network.RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device))
            if os.path.isfile('Networks/'+market+'/Seed'+str(i)+'.pt'):
                self.models[i].load_state_dict(torch.load('Networks/'+market+'/Seed'+str(i)+'.pt'))
            self.optimisers.append(torch.optim.Adam(self.models[i].parameters(), lr=0.01))
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler2 = MinMaxScaler(feature_range=(-1, 1))
            if not os.path.isfile('Networks/'+market+'/Seed'+str(i)+'.pt'):
                self.__trainInitial(i)
    
    def __save(self, i):
        os.makedirs('Networks/'+self.market, exist_ok=True)
        torch.save(self.models[i].state_dict(), 'Networks/'+self.market+'/Seed'+str(i)+'.pt')
    
    def __loadData(self):
        #self.dataHistoDict = np.load('../CoinbaseTrade/HistoricalData/historicalBTC-EUR.npy', allow_pickle=True)
        last = self.dataHistoDict[1]
        ask = self.dataHistoDict[2]
        bid = self.dataHistoDict[3]
        askVolume = self.dataHistoDict[4]
        bidVolume = self.dataHistoDict[5]

        last = self.scaler.fit_transform(last.reshape(-1, 1))
        ask = self.scaler.transform(ask.reshape(-1, 1))
        bid = self.scaler.transform(bid.reshape(-1, 1))
        askVolume = self.scaler2.fit_transform(askVolume.reshape(-1, 1))
        bidVolume = self.scaler2.transform(bidVolume.reshape(-1, 1))

        xtrain1, ytrain1 = self.__load_data(last, look_back)
        xtrain2, ytrain2 = self.__load_data(ask, look_back)
        xtrain3, ytrain3 = self.__load_data(bid, look_back)
        xtrain4, ytrain4 = self.__load_data(askVolume, look_back)
        xtrain5, ytrain5 = self.__load_data(bidVolume, look_back)

        # make training and test sets in torch
        xtrain1 = torch.from_numpy(xtrain1).type(torch.Tensor).to(device)
        xtrain2 = torch.from_numpy(xtrain2).type(torch.Tensor).to(device)
        xtrain3 = torch.from_numpy(xtrain3).type(torch.Tensor).to(device)
        xtrain4 = torch.from_numpy(xtrain4).type(torch.Tensor).to(device)
        xtrain5 = torch.from_numpy(xtrain5).type(torch.Tensor).to(device)
        ytrain2 = torch.from_numpy(ytrain2).type(torch.Tensor).to(device)
        ytrain3 = torch.from_numpy(ytrain3).type(torch.Tensor).to(device)

        self.inputTrain = torch.cat((xtrain1, xtrain2, xtrain3, xtrain4, xtrain5), -1)
        self.outputTrain = torch.cat((ytrain2, ytrain3), -1)

    # function to create train, test data given stock data and sequence length
    def __load_data(self, stock, look_back):
        data_raw = stock  # convert to numpy array
        data = []

        # create all possible sequences of length look_back
        for index in range(len(data_raw) - look_back):
            if self.dataHistoDict[0][index+look_back] - self.dataHistoDict[0][index] == look_back:
                data.append(data_raw[index: index + look_back])

        data = np.array(data)

        x_train = data[:, :-8, :]
        y_train = data[:, -1, :]

        return [x_train, y_train]
    
    def __trainInitial(self, i, num_epochs=250):
        self.__loadData()
        loss_fn = torch.nn.MSELoss()
        # Train model
        #####################
        hist = np.zeros(num_epochs)

        # Number of steps to unroll
        seq_dim = look_back-1

        for t in range(num_epochs):
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            #model.hidden = model.init_hidden()

            # Forward pass
            ytrainpred = self.models[i](self.inputTrain)

            loss = loss_fn(ytrainpred, self.outputTrain)
            # if t % 10 == 0 and t != 0:
            #     print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            self.optimisers[i].zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimisers[i].step()
        self.__save(i)
    
    def train(self, num_epochs=100):
        for i in range(num_seeds):
            self.__loadData()
            loss_fn = torch.nn.MSELoss()
            # Train model
            #####################
            hist = np.zeros(num_epochs)

            # Number of steps to unroll
            seq_dim = look_back-1

            for t in range(num_epochs):
                # Initialise hidden state
                # Don't do this if you want your LSTM to be stateful
                #model.hidden = model.init_hidden()

                # Forward pass
                ytrainpred = self.models[i](self.inputTrain)

                loss = loss_fn(ytrainpred, self.outputTrain)
                # if t % 10 == 0 and t != 0:
                #     print("Epoch ", t, "MSE: ", loss.item())
                hist[t] = loss.item()

                # Zero out gradient, else they will accumulate between epochs
                self.optimisers[i].zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimisers[i].step()
            self.__save(i)
    
    def predict(self, data):
        #self.dataHistoDict = np.load('../CoinbaseTrade/HistoricalData/historicalBTC-EUR.npy', allow_pickle=True)
        self.dataHistoDict = data
        last = self.dataHistoDict[1]
        print('length of data'+str(len(last)))
        ask = self.dataHistoDict[2]
        bid = self.dataHistoDict[3]
        askVolume = self.dataHistoDict[4]
        bidVolume = self.dataHistoDict[5]
        last = self.scaler.fit_transform(last.reshape(-1, 1))
        ask = self.scaler.transform(ask.reshape(-1, 1))
        bid = self.scaler.transform(bid.reshape(-1, 1))
        askVolume = self.scaler2.fit_transform(askVolume.reshape(-1, 1))
        bidVolume = self.scaler2.transform(bidVolume.reshape(-1, 1))
        asks = []
        bids = []
        patternlast = []
        patternask = []
        patternbid = []
        patternaskvolume = []
        patternbidvolume = []
        patternlast.append(last[-(look_back-5):])
        patternask.append(ask[-(look_back-5):])
        patternbid.append(bid[-(look_back-5):])
        patternaskvolume.append(askVolume[-(look_back-5):])
        patternbidvolume.append(bidVolume[-(look_back-5):])
        patternlast = np.array(patternlast)
        patternask = np.array(patternask)
        patternbid = np.array(patternbid)
        patternaskvolume = np.array(patternaskvolume)
        patternbidvolume = np.array(patternbidvolume)
        patternlast = torch.from_numpy(patternlast).type(torch.Tensor).to(device)
        patternask = torch.from_numpy(patternask).type(torch.Tensor).to(device)
        patternbid = torch.from_numpy(patternbid).type(torch.Tensor).to(device)
        patternaskvolume = torch.from_numpy(patternaskvolume).type(torch.Tensor).to(device)
        patternbidvolume = torch.from_numpy(patternbidvolume).type(torch.Tensor).to(device)
        pattern = torch.cat((patternlast, patternask, patternbid, patternaskvolume, patternbidvolume), -1)
        for i in range(num_seeds):
            self.models[i].eval()
            yprediction = self.models[i](pattern)
            yprediction = self.scaler.inverse_transform(yprediction.detach().cpu().numpy())
            asks.append(yprediction[0][0])
            bids.append(yprediction[0][1])
            self.models[i].train()
        asks = np.array(asks)
        bids = np.array(bids)
        return asks.mean(), bids.mean(), asks.std(), bids.std()