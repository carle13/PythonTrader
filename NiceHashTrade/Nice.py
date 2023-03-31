from hmac import new
from io import StringIO
from NiceHashTrade import nicehash
import numpy as np
from os import system
import json
import threading
import time
from prettytable import PrettyTable
import config
import common
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from ArtificialIntelligenceTest import networkManager

def getData(market):
	global lastPrices
	global ask
	global bid
	global askVolume
	global bidVolume
	# Get ask and bid
	try:
		sp = public_api.get_exchange_orderbook(market, 1)
	except:
		print('Error when getting orderbook'+market)
		return False
	if 'sell' in sp and 'buy' in sp:
		bid[market] = float(sp['sell'][0][0])
		bidVolume[market] = float(sp['sell'][0][1])
		ask[market] = float(sp['buy'][0][0])
		askVolume[market] = float(sp['buy'][0][1])
	else:
		print(sp)
		removeMarket(market)
		return False
	return True

def saveData():
	global btcPairs
	global lastPrices
	global currentTime
	global dataHistoDict
	global dataTrans
	# Update file data
	file = open("NiceHashTrade/buyAllowed.json", "w")
	json.dump(buyAllowed, file, sort_keys=True)
	file.close()

	# Save historical data into different csv files
	for i in range(len(btcPairs)):
		market = btcPairs[i]['symbol']
		if market not in lastPrices:
			# # Get exchange market info
			# exchange_info = public_api.get_exchange_markets_info()
			# pairsBitcoin = list(exchange_info['symbols'][i] for i in range(len(exchange_info['symbols'])) if exchange_info['symbols'][i]['quoteAsset'] == 'BTC' and exchange_info['symbols'][i]['status'] == 'TRADING')
			# # Get indices for market symbols
			# indexMarket = dict()
			# for i in range(len(pairsBitcoin)):
			# 	indexMarket[pairsBitcoin[i]['symbol']] = i
			continue

		np.save('NiceHashTrade/HistoricalData/historical' + market +'.npy', dataHistoDict[market])
		if len(dataTrans[market][0]) > 1:
			if dataTrans[market][0][0] < dataHistoDict[market][0][0]:
				dataTrans[market] = [dataTrans[market][0][1:], dataTrans[market][1][1:], dataTrans[market][2][1:]]
		np.save('NiceHashTrade/TransactionData/transactions' + market + '.npy', dataTrans[market])


def getTable(markets, changePurchase, holding, changePrice, spread):
	markets.clear()
	changePurchase.clear()
	holding.clear()
	changePrice.clear()
	spread.clear()
	for i in range(len(btcPairs)):
		key = btcPairs[i]['symbol']
		markets.append(key)
		if key in buyAllowed:
			holding.append(not buyAllowed[key])
		else:
			holding.append(False)
		if key in changeLastPurchase:
			changePurchase.append(changeLastPurchase[key])
			changePrice.append(changeLastPrice[key])
			spread.append(spreadCurrencies[key])
		else:
			changePurchase.append(0.0)
			changePrice.append(0.0)
			spread.append(0.0)

def plotBase(market, ax):
	points = 100
	ax.set_title('Market: ' + market)
	ax.set_xlabel('Time')
	ax.set_ylabel('Price')
	if market not in dataHistoDict:
		return
	ax.plot(dataHistoDict[market][0][-points:], dataHistoDict[market][1][-points:], label='Last Price')
	# ax.plot(dataHistoDict[market][0][-len(shortAverage[market]):][-points:], shortAverage[market][-points:], label='Short Average')
	# ax.plot(dataHistoDict[market][0][-len(mediumAverage[market]):][-points:], mediumAverage[market][-points:], label='Medium Average')
	# ax.plot(dataHistoDict[market][0][-len(longAverage[market]):][-points:], longAverage[market][-points:], label='Long Average')
	ax.plot(dataHistoDict[market][0][-points:], dataHistoDict[market][2][-points:], label='Ask')
	ax.plot(dataHistoDict[market][0][-points:], dataHistoDict[market][3][-points:], label='Bid')
	if market in dataTrans:
		buy = [[], []]
		sell = [[], []]
		buy[0] = list(dataTrans[market][0][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 'b' and currentTime - dataTrans[market][0][i] <= points)
		buy[1] = list(dataTrans[market][1][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 'b' and currentTime - dataTrans[market][0][i] <= points)
		sell[0] = list(dataTrans[market][0][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 's' and currentTime - dataTrans[market][0][i] <= points)
		sell[1] = list(dataTrans[market][1][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 's' and currentTime - dataTrans[market][0][i] <= points)
		ax.plot(buy[0], buy[1], 'o', label='Buys')
		ax.plot(sell[0], sell[1], 'o', label='Sells')
	if market in predictions:
		ax.errorbar(predictions[market][0][:], predictions[market][1][:], yerr=predictions[market][3][:], label='Prediction Ask', capsize=5.0)
		ax.errorbar(predictions[market][0][:], predictions[market][2][:], yerr=predictions[market][4][:], label='Prediction Bid', capsize=5.0)
	ax.legend()

def newMarket(market):
	global btcPairs
	global lastPrices
	global ask
	global bid
	global askVolume
	global bidVolume
	global indexMarket
	global indexCurrency
	global currentTime
	print('New market ' + str(market))
	if market not in buyAllowed:
		buyAllowed[market] = True
	dataTrans[market] = np.array([np.array([], dtype=type(currentTime)), np.array([], dtype=type(lastPrices[market])), np.array([], dtype=type('s'))], dtype=object)
	dataHistoDict[market] = np.array([np.array([currentTime]), np.array(
		[lastPrices[market]]), np.array([ask[market]]), np.array([bid[market]]), np.array([askVolume[market]]), np.array([bidVolume[market]])], dtype=object)

def removeMarket(market):
	global btcPairs
	global lastPrices
	global ask
	global bid
	global askVolume
	global bidVolume
	global indexMarket
	global indexCurrency
	print('Removing market ' +str(market))
	try:
		btcPairs.pop(indexMarket[market])
	except:
		print('Market was not in btcPairs')
	lastPrices.pop(market, 0)
	ask.pop(market, 0)
	bid.pop(market, 0)
	askVolume.pop(market, 0)
	bidVolume.pop(market, 0)

def buy(market):
	global dataTrans
	global buyAllowed
	global ordersCheck
	global currentTime
	global ask
	global bid
	global transactionsAttempted
	global transactionsDone
	quoteAsset = btcPairs[indexMarket[market]]['quoteAsset']
	available = float(my_accounts['currencies'][indexCurrency[quoteAsset]]['available'])
	minQuote = float(exchange_info['symbols'][indexMarket[market]]['secMinAmount'])

	if available > 0.0001 and available != 0.0:
		# Create buy market order
		print('Buying: ' + market + ' ____ Amount: ' + str(0.0001))
		try:
			new_buy_market_order = private_api.create_exchange_buy_market_order(market, 0.0001)
			if new_buy_market_order['state'] == 'FULL':
				dataTrans[market] = [np.append(dataTrans[market][0], currentTime), np.append(dataTrans[market][1], float(new_buy_market_order['price'])), np.append(dataTrans[market][2], 'b')]
				buyAllowed[market] = False
				# Add successful tradde
				transactionsDone += 1
			else:
				marketsCheck.append(market)
		except Exception as inst:
			# Free lock to release next thread
			print(inst)
			return

	else:
		# Unsuccessful trade
		transactionsAttempted += 1

def sell(market):
	global dataTrans
	global buyAllowed
	global ordersCheck
	global currentTime
	global ask
	global bid
	global transactionsAttempted
	global transactionsDone
	baseAsset = btcPairs[indexMarket[market]]['baseAsset']
	available = float(my_accounts['currencies'][indexCurrency[baseAsset]]['available'])
	minBase = float(btcPairs[indexMarket[market]]['priMinAmount'])

	if available > minBase and available != 0.0:
		# Create sell market order
		print('Selling: ' + market + ' ____ Amount: ' + str(available))
		try:
			new_sell_market_order = private_api.create_exchange_sell_market_order(
				market, available)
			if new_sell_market_order['state'] == 'FULL':
				dataTrans[market] = [np.append(dataTrans[market][0], currentTime), np.append(dataTrans[market][1], float(new_sell_market_order['price'])), np.append(dataTrans[market][2], 's')]
				buyAllowed[market] = True
				# Add successful tradde
				transactionsDone += 1
			else:
				marketsCheck.append(market)
		except Exception as inst:
			# Free lock to release next thread
			print(inst)
			return
	else:
		# Unsuccessful trade
		transactionsAttempted += 1

host = config.nicehost
organisation_id = config.niceorganisation_id
key = config.nicekey
secret = config.nicesecret

# Variable initialization
transactionsDone = 0
transactionsAttempted = 0
transactionsCancelled = 0
currentTime = 0.0

marketsCheck = []

shortPeriod = 2
mediumPeriod = 10
longPeriod = 48

threadLock = threading.Lock()

print('################################')
print('Initializing variables NiceHash')

############################################
# PUBLIC FUNCTIONS
# Create public api object
public_api = nicehash.public_api(host, False)


############################################
# PRIVATE FUNCTIONS
# Create private api object
private_api = nicehash.private_api(host, organisation_id, key, secret, False)

# Get balance for all currencies
my_accounts = private_api.get_accounts()


############################################
# EXCHANGE
# Get exchange market info
exchange_info = public_api.get_exchange_markets_info()
btcPairs = list(exchange_info['symbols'][i] for i in range(len(exchange_info['symbols'])) if exchange_info['symbols'][i]['quoteAsset'] == 'BTC' and exchange_info['symbols'][i]['status'] == 'TRADING')

# Get last prices
lastPrices = dict()
# lastPrices = public_api.get_exchange_lastPrices()
ask = dict()
bid = dict()
askVolume = dict()
bidVolume = dict()


#########################################
# TRADING
# Read file data to see which currencies are currently on hold
try:
	file = open('NiceHashTrade/buyAllowed.json', 'r')
	buyAllowed = json.load(file)
	file.close()
except:
	buyAllowed = dict()
	for i in range(len(btcPairs)):
		market = btcPairs[i]['symbol']
		buyAllowed[market] = True

# Read historical prices for the currencies
dataHistoDict = dict()
shortAverage = dict()
mediumAverage = dict()
longAverage = dict()
for i in range(len(btcPairs)):
	market = btcPairs[i]['symbol']
	try:
		dataHistoDict[market] = np.load('NiceHashTrade/HistoricalData/historical' + market + '.npy', allow_pickle=True)
		shortAverage[market] = common.calculate_ema(dataHistoDict[market][1], shortPeriod)
		mediumAverage[market] = common.calculate_ema(dataHistoDict[market][1], mediumPeriod)
		longAverage[market] = common.calculate_ema(dataHistoDict[market][1], longPeriod)
	except:
		print('No data for market: ' + market)

#Read transaction history
dataTrans = dict()
for i in range(len(btcPairs)):
	market = btcPairs[i]['symbol']
	try:
		dataTrans[market] = np.load('NiceHashTrade/TransactionData/transactions' + market + '.npy', allow_pickle=True)
	except:
		print('No transactions recorded before: ' + market)

# Read the change since last purchase
changeLastPurchase = dict()

# Read change since last time it was checked
changeLastPrice = dict()

# Variable to calculate the spread of the orderbook
spreadCurrencies = dict()

# Create dictionary to store neural network predictions
predictions = dict()

# Create dictionary to hold manager of networks
networks = dict()

# Get indices for market symbols
indexMarket = dict()
for i in range(len(btcPairs)):
	indexMarket[btcPairs[i]['symbol']] = i

# Get indices for accounts
indexCurrency = dict()
for i in range(len(my_accounts['currencies'])):
	indexCurrency[my_accounts['currencies'][i]['currency']] = i


# Thread class
class niceThread(threading.Thread):
	def __init__(self, threadID, name):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name

	def run(self):
		global transactionsAttempted
		global transactionsDone
		global my_accounts
		global lastPrices
		global ask
		global bid
		global askVolume
		global bidVolume
		global buyAllowed
		global dataHistoDict
		global spreadCurrencies
		global changeLastPurchase
		global changeLastPrice
		global dataTrans
		global shortAverage
		global mediumAverage
		global longAverage
		global marketsCheck
		market = self.name
		# Get lock to synchronize threads
		threadLock.acquire()

		# Last price checking done in the main function
		# Get bid, ask and volumes
		if not getData(market):
			# Free lock and return
			threadLock.release()
			return

		# Check if the market is newly added
		if not market in dataHistoDict:
			newMarket(market)
		
		# Append data to historical price data
		dataHistoDict[market] = [np.append(dataHistoDict[market][0], currentTime), np.append(
			dataHistoDict[market][1], lastPrices[market]), np.append(dataHistoDict[market][2], ask[market]), np.append(dataHistoDict[market][3], bid[market]), np.append(dataHistoDict[market][4], askVolume[market]), np.append(dataHistoDict[market][5], bidVolume[market])]
		dataStore = 2000
		if len(dataHistoDict[market][1]) > dataStore:
			dataHistoDict[market] = [dataHistoDict[market][0][-dataStore:],
                dataHistoDict[market][1][-dataStore:], dataHistoDict[market][2][-dataStore:], dataHistoDict[market][3][-dataStore:], dataHistoDict[market][4][-dataStore:], dataHistoDict[market][5][-dataStore:]]
		shortAverage[market] = common.calculate_ema(dataHistoDict[market][1], shortPeriod)
		mediumAverage[market] = common.calculate_ema(dataHistoDict[market][1], mediumPeriod)
		longAverage[market] = common.calculate_ema(dataHistoDict[market][1], longPeriod)

		# Calculate spread
		spreadCurrencies[market] = (abs(bid[market] - ask[market]) / lastPrices[market]) * 100

		# Get change since last purchase
		if len(dataTrans[market][1]) > 0:
			changeLastPurchase[market] = ((lastPrices[market] - dataTrans[market][1][-1]) / dataTrans[market][1][-1]) * 100.0
		else:
			changeLastPurchase[market] = 0.0

		#Get change since last 10 prices
		if len(dataHistoDict[market][1]) > 10:
			previousPrice = dataHistoDict[market][1][-10]
		else:
			previousPrice = dataHistoDict[market][1][-1]
		changeLastPrice[market] = ((lastPrices[market] - previousPrice) / previousPrice) * 100.0

		# check that is not a low fluidity pair
		if spreadCurrencies[market] >= 0.30:
			threadLock.release()
			return

		# check that there is continuous data
		if len(dataHistoDict[market][0]) >= longPeriod+1:
			if dataHistoDict[market][0][-1] - dataHistoDict[market][0][-(longPeriod+1)] > longPeriod+1:
				threadLock.release()
				return
		else:
			threadLock.release()
			return

		# machine learning
		if market not in networks:
			try:
				networks[market] = networkManager.NeuralNetwork(market, dataHistoDict[market])
			except Exception as inst:
				print('Error when creating network for '+market)
				print(inst)
				threadLock.release()
				return

		# Get predictions from the network
		askpred, bidpred, askdev, biddev = networks[market].predict(dataHistoDict[market])
		if market in predictions:
			predictions[market] = [np.append(predictions[market][0], currentTime+7), np.append(
				predictions[market][1], askpred), np.append(predictions[market][2], bidpred), np.append(predictions[market][3], askdev), np.append(predictions[market][4], biddev)]
		else:
			predictions[market] = np.array([np.array([currentTime+7]), np.array(
				[askpred]), np.array([bidpred]), np.array([askdev]), np.array([biddev])], dtype=object)
		
		dataPreds = 48
		if len(predictions[market][1]) > dataPreds:
			predictions[market] = [predictions[market][0][-dataPreds:],
				predictions[market][1][-dataPreds:], predictions[market][2][-dataPreds:], predictions[market][3][-dataPreds:], predictions[market][4][-dataPreds:]]

		# If market goes up sell
		if mediumAverage[market][-1] > shortAverage[market][-1] and mediumAverage[market][-2] <= shortAverage[market][-2] and not buyAllowed[market] and mediumAverage[market][-1] >= longAverage[market][-1]: #  and changeLastPurchase[market] > 3.0
			# Check if there is an account for the base currency
			baseAsset = btcPairs[indexMarket[market]]['baseAsset']
			if not baseAsset in indexCurrency:
				print('Currency not available: ' + market)
				threadLock.release()
				return
			sell(market)

		# If market goes down buy
		if longAverage[market][-1] < shortAverage[market][-1] and longAverage[market][-2] > shortAverage[market][-2] and longAverage[market][-1] >= mediumAverage[market][-1] and buyAllowed[market]:
			# Check if there is an account for quote currency
			quoteAsset = btcPairs[indexMarket[market]]['quoteAsset']
			if not quoteAsset in indexCurrency:
				print('Currency not available: ' + market)
				threadLock.release()
				return
			#buy(market)

		# Free lock to release next thread
		threadLock.release()
		return


# Get balance for all currencies
my_accounts = private_api.get_accounts()
# Get last prices
lastPrices = public_api.get_exchange_lastPrices()

def trade(t, markets, changePurchase, holding, changePrice, spread, transactions, ini=False):
	global currentTime
	currentTime = t
	global btcPairs
	global transactionsDone
	global transactionsCancelled
	global my_accounts
	global lastPrices
	global marketsCheck
	global predictions
	l = len(btcPairs)+3
	common.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 25)

	marketsCheck = []

	try:
		# Get balance for all currencies
		my_accounts = private_api.get_accounts()
		# Get indices for accounts
		indexCurrency = dict()
		for i in range(len(my_accounts['currencies'])):
			indexCurrency[my_accounts['currencies'][i]['currency']] = i
		# Get last prices
		lastPrices = public_api.get_exchange_lastPrices()
	except:
		print('Error when getting data from server')
		return

	if ini:
		getTable(markets, changePurchase, holding, changePrice, spread)
		return

	threads = []

	# Initialize threads
	for i in range(len(btcPairs)):
		market = btcPairs[i]['symbol']
		threads.append(niceThread(i, market))

	for t in threads:
		t.start()

	i = 0
	for t in threads:
		try:
			t.join()
		except Exception as inst:
			print('Error in thread')
			print(inst)
			progreso = common.printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=25)
			i += 1
			continue
		progreso = common.printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=25)
		i += 1
		transactions.clear()
		# Print information
		transactions.append('Transactions done: ' + str(transactionsDone) + '\nTransactions attempted: ' + str(
			transactionsAttempted) + '\nTransactions cancelled: ' + str(transactionsCancelled)
			+ '\n' + progreso)
	getTable(markets, changePurchase, holding, changePrice, spread)

	time.sleep(5)

	for check in marketsCheck:
		my_exchange_orders = private_api.get_my_exchange_orders(check)
		if my_exchange_orders[0]['state'] == 'FULL':
			if my_exchange_orders[0]['side'] == 'BUY':
				buyAllowed[check] = False
				dataTrans[check] = [np.append(dataTrans[check][0], currentTime), np.append(dataTrans[check][1], lastPrices[check]), np.append(dataTrans[check][2], 'b')]
			else:
				buyAllowed[check] = True
				dataTrans[check] = [np.append(dataTrans[check][0], currentTime), np.append(dataTrans[check][1], lastPrices[check]), np.append(dataTrans[check][2], 's')]
			# Add successful trade
			transactionsDone += 1
			print('transaction finished')
		else:
			print('transaction being cancelled')
			transactionsCancelled += 1
			cancelled_order = private_api.cancel_exchange_order(check, my_exchange_orders[0]['orderId'])
	progreso = common.printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=25)
	i += 1
	transactions.clear()
	# Print information
	transactions.append('Transactions done: ' + str(transactionsDone) + '\nTransactions attempted: ' + str(
		transactionsAttempted) + '\nTransactions cancelled: ' + str(transactionsCancelled)
		+ '\n' + progreso)
	saveData()
	progreso = common.printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=25)
	i += 1
	transactions.clear()
	# Print information
	transactions.append('Transactions done: ' + str(transactionsDone) + '\nTransactions attempted: ' + str(
		transactionsAttempted) + '\nTransactions cancelled: ' + str(transactionsCancelled)
		+ '\n' + progreso)
	# Perform continual training of networks
	for key in networks:
		networks[key].train()
	progreso = common.printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=25)
	i += 1
	transactions.clear()
	# Print information
	transactions.append('Transactions done: ' + str(transactionsDone) + '\nTransactions attempted: ' + str(
		transactionsAttempted) + '\nTransactions cancelled: ' + str(transactionsCancelled)
		+ '\n' + progreso)
