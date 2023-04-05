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
input_dim = 1
hidden_dim = 50
num_layers = 2
output_dim = 1
num_seeds = 4
look_back = 48  # choose sequence length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork():
    def __init__(self, market, data):
        self.market = market
        self.dataHistoDict = data
        self.modelsAsk = []
        self.modelsBid = []
        self.optimisersAsk = []
        self.optimisersBid = []
        self.askScaler = MinMaxScaler(feature_range=(-1, 1))
        self.bidScaler = MinMaxScaler(feature_range=(-1, 1))
        for i in range(num_seeds):
            #Network for ask prices
            self.modelsAsk.append(network.RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device))
            if os.path.isfile('Networks/'+market+'/askSeed'+str(i)+'.pt'):
                self.modelsAsk[i].load_state_dict(torch.load('Networks/'+market+'/askSeed'+str(i)+'.pt'))
            self.optimisersAsk.append(torch.optim.Adam(self.modelsAsk[i].parameters(), lr=0.01))
            #Network for bid prices
            self.modelsBid.append(network.RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device))
            if os.path.isfile('Networks/'+market+'/bidSeed'+str(i)+'.pt'):
                self.modelsBid[i].load_state_dict(torch.load('Networks/'+market+'/bidSeed'+str(i)+'.pt'))
            self.optimisersBid.append(torch.optim.Adam(self.modelsBid[i].parameters(), lr=0.01))
            #If networks were not created before train them for first time
            if not os.path.isfile('Networks/'+market+'/bidSeed'+str(i)+'.pt') or not os.path.isfile('Networks/'+market+'/askSeed'+str(i)+'.pt'):
                print('Train initial called')
                self.__trainInitial(i)
    
    def __save(self, i):
        os.makedirs('Networks/'+self.market, exist_ok=True)
        torch.save(self.modelsAsk[i].state_dict(), 'Networks/'+self.market+'/askSeed'+str(i)+'.pt')
        torch.save(self.modelsBid[i].state_dict(), 'Networks/'+self.market+'/bidSeed'+str(i)+'.pt')
    
    def __loadData(self):
        #self.dataHistoDict = np.load('../CoinbaseTrade/HistoricalData/historicalBTC-EUR.npy', allow_pickle=True)
        ask = self.dataHistoDict[2]
        bid = self.dataHistoDict[3]

        ask = self.askScaler.fit_transform(ask.reshape(-1, 1))
        bid = self.bidScaler.fit_transform(bid.reshape(-1, 1))

        xtrainAsk, ytrainAsk = self.__load_data(ask, look_back)
        xtrainBid, ytrainBid = self.__load_data(bid, look_back)

        # make training and test sets in torch
        self.xtrainAsk = torch.from_numpy(xtrainAsk).type(torch.Tensor).to(device)
        self.xtrainBid = torch.from_numpy(xtrainBid).type(torch.Tensor).to(device)
        self.ytrainAsk = torch.from_numpy(ytrainAsk).type(torch.Tensor).to(device)
        self.ytrainBid = torch.from_numpy(ytrainBid).type(torch.Tensor).to(device)

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
        # Train model for ask prices
        #####################
        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            # Forward pass
            ytrainpredAsk = self.modelsAsk[i](self.xtrainAsk)
            loss = loss_fn(ytrainpredAsk, self.ytrainAsk)
            hist[t] = loss.item()
            # Zero out gradient, else they will accumulate between epochs
            self.optimisersAsk[i].zero_grad()
            # Backward pass
            loss.backward()
            # Update parameters
            self.optimisersAsk[i].step()
        # Train model for bid prices
        #####################
        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            # Forward pass
            ytrainpredBid = self.modelsBid[i](self.xtrainBid)
            loss = loss_fn(ytrainpredBid, self.ytrainBid)
            hist[t] = loss.item()
            # Zero out gradient, else they will accumulate between epochs
            self.optimisersBid[i].zero_grad()
            # Backward pass
            loss.backward()
            # Update parameters
            self.optimisersBid[i].step()
        self.__save(i)
    
    def train(self, num_epochs=100):
        self.__loadData()
        loss_fn = torch.nn.MSELoss()
        for i in range(num_seeds):
            # Train model for ask prices
            #####################
            hist = np.zeros(num_epochs)
            for t in range(num_epochs):
                # Forward pass
                ytrainpredAsk = self.modelsAsk[i](self.xtrainAsk)
                loss = loss_fn(ytrainpredAsk, self.ytrainAsk)
                hist[t] = loss.item()
                # Zero out gradient, else they will accumulate between epochs
                self.optimisersAsk[i].zero_grad()
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimisersAsk[i].step()
            # Train model for bid prices
            #####################
            hist = np.zeros(num_epochs)
            for t in range(num_epochs):
                # Forward pass
                ytrainpredBid = self.modelsBid[i](self.xtrainBid)
                loss = loss_fn(ytrainpredBid, self.ytrainBid)
                hist[t] = loss.item()
                # Zero out gradient, else they will accumulate between epochs
                self.optimisersBid[i].zero_grad()
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimisersBid[i].step()
            self.__save(i)
    
    def predict(self, data):
        #self.dataHistoDict = np.load('../CoinbaseTrade/HistoricalData/historicalBTC-EUR.npy', allow_pickle=True)
        self.dataHistoDict = data
        ask = self.dataHistoDict[2]
        bid = self.dataHistoDict[3]
        ask = self.askScaler.fit_transform(ask.reshape(-1, 1))
        bid = self.bidScaler.fit_transform(bid.reshape(-1, 1))
        asks = []
        bids = []
        patternask = []
        patternbid = []
        patternask.append(ask[-(look_back-5):])
        patternbid.append(bid[-(look_back-5):])
        patternask = np.array(patternask)
        patternbid = np.array(patternbid)
        patternask = torch.from_numpy(patternask).type(torch.Tensor).to(device)
        patternbid = torch.from_numpy(patternbid).type(torch.Tensor).to(device)
        for i in range(num_seeds):
            self.modelsAsk[i].eval()
            self.modelsBid[i].eval()
            yAsk = self.modelsAsk[i](patternask)
            yBid = self.modelsBid[i](patternbid)
            yAsk = self.askScaler.inverse_transform(yAsk.detach().cpu().numpy())
            yBid = self.bidScaler.inverse_transform(yBid.detach().cpu().numpy())
            asks.append(yAsk)
            bids.append(yBid)
            self.modelsAsk[i].train()
            self.modelsBid[i].train()
        asks = np.array(asks)
        bids = np.array(bids)
        return asks.mean(), bids.mean(), asks.std(), bids.std()