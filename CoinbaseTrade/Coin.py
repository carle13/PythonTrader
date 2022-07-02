from io import StringIO
import cbpro
import base64
from cbpro import public_client
import numpy as np
from os import system
import json
import threading
import time
from prettytable import PrettyTable
import config
import common
import matplotlib.pyplot as plt

def saveData():
	global euroPairs
	global lastPrices
	global currentTime
	global dataHistoDict
	global dataTrans
	# Update file data
	file = open("CoinbaseTrade/buyAllowed.json", "w")
	json.dump(buyAllowed, file, sort_keys=True)
	file.close()

	# Save historical data into different csv files
	for i in range(len(euroPairs)):
		market = euroPairs[i]['id']
		if market not in lastPrices:
			# Get exchange market info EUROS
			pairs = c.get_products()
			euroPairs = list(pairs[i] for i in range(len(pairs)) if pairs[i]['quote_currency'] == 'EUR' and not pairs[i]['trading_disabled'])
			# Get indices for market symbols
			indexMarket = dict()
			for i in range(len(euroPairs)):
				indexMarket[euroPairs[i]['id']] = i
			continue

		np.save('CoinbaseTrade/HistoricalData/historical' + market +'.npy', dataHistoDict[market])

		if dataTrans[market][0][0] < dataHistoDict[market][0][0]:
			dataTrans[market] = [dataTrans[market][0][1:], dataTrans[market][1][1:], dataTrans[market][2][1:]]
		np.save('CoinbaseTrade/TransactionData/transactions' + market + '.npy', dataTrans[market])


def getTable(markets, changePurchase, holding, changePrice, spread):
	markets.clear()
	changePurchase.clear()
	holding.clear()
	changePrice.clear()
	spread.clear()
	for i in range(len(euroPairs)):
		key = euroPairs[i]['id']
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
	ax.set_title('Market: ' + market)
	ax.set_xlabel('Time')
	ax.set_ylabel('Price')
	if market not in dataHistoDict:
		return
	ax.plot(dataHistoDict[market][0], dataHistoDict[market][1], label='Last Price')
	ax.plot(dataHistoDict[market][0][-len(shortAverage[market]):], shortAverage[market], label='Short Average')
	ax.plot(dataHistoDict[market][0][-len(mediumAverage[market]):], mediumAverage[market], label='Medium Average')
	ax.plot(dataHistoDict[market][0][-len(longAverage[market]):], longAverage[market], label='Long Average')
	if market not in dataTrans:
		return
	buy = [[], []]
	sell = [[], []]
	buy[0] = list(dataTrans[market][0][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 'b')
	buy[1] = list(dataTrans[market][1][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 'b')
	sell[0] = list(dataTrans[market][0][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 's')
	sell[1] = list(dataTrans[market][1][i] for i in range(len(dataTrans[market][0])) if dataTrans[market][2][i] == 's')
	ax.plot(buy[0], buy[1], 'o', label='Buys')
	ax.plot(sell[0], sell[1], 'o', label='Sells')
	ax.legend()

key = config.coinkey
secret = config.coinsecret
passphrase = config.coinpassphrase

# Variable initialization
transactionsDone = 0
transactionsAttempted = 0
transactionsCancelled = 0
currentTime = 0.0

ordersCheck = []

shortPeriod = 2
mediumPeriod = 10
longPeriod = 20

threadLock = threading.Lock()

print('################################')
print('Initializing variables CoinBase')

############################################
# PUBLIC FUNCTIONS
# Create public api object
c = cbpro.PublicClient()


############################################
# PRIVATE FUNCTIONS
# Create private api object
encoded = json.dumps(secret).encode()
b64secret = base64.b64encode(encoded)
auth_client = cbpro.AuthenticatedClient(key=key, b64secret=secret, passphrase=passphrase)

# Get balance for all currencies
my_accounts = auth_client.get_accounts()


############################################
# EXCHANGE
# Get exchange market info EUROS
pairs = c.get_products()
euroPairs = list(pairs[i] for i in range(len(pairs)) if pairs[i]['quote_currency'] == 'EUR' and not pairs[i]['trading_disabled'])

# Get last prices
lastPrices = dict()
ask = dict()
bid = dict()
askVolume = dict()
bidVolume = dict()


#########################################
# TRADING
# Read file data to see which currencies are currently on hold
try:
	file = open('CoinbaseTrade/buyAllowed.json', 'r')
	buyAllowed = json.load(file)
	file.close()
except:
	buyAllowed = dict()
	for i in range(len(euroPairs)):
		market = euroPairs[i]['id']
		buyAllowed[market] = True

# Read historical prices for the currencies
dataHistoDict = dict()
shortAverage = dict()
mediumAverage = dict()
longAverage = dict()
for i in range(len(euroPairs)):
	market = euroPairs[i]['id']
	try:
		dataHistoDict[market] = np.load('CoinbaseTrade/HistoricalData/historical' + market + '.npy', allow_pickle=True)
		shortAverage[market] = common.calculate_ema(dataHistoDict[market][1], shortPeriod)
		mediumAverage[market] = common.calculate_ema(dataHistoDict[market][1], mediumPeriod)
		longAverage[market] = common.calculate_ema(dataHistoDict[market][1], longPeriod)
	except:
		print('New market: ' + market)

#Read transaction history
dataTrans = dict()
for i in range(len(euroPairs)):
	market = euroPairs[i]['id']
	try:
		dataTrans[market] = np.load('CoinbaseTrade/TransactionData/transactions' + market + '.npy', allow_pickle=True)
	except:
		print('No transactions recorded before: ' + market)

# Read the change since last purchase
changeLastPurchase = dict()

# Read change since last time it was checked
changeLastPrice = dict()

# Variable to calculate the spread of the orderbook
spreadCurrencies = dict()

# Get indices for market symbols
indexMarket = dict()
for i in range(len(euroPairs)):
	indexMarket[euroPairs[i]['id']] = i

# Get indices for accounts
indexCurrency = dict()
for i in range(len(my_accounts)):
	indexCurrency[my_accounts[i]['currency']] = i


# Thread class
class coinThread(threading.Thread):
	def __init__(self, threadID, name):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name

	def run(self):
		global transactionsAttempted
		global transactionsDone
		global my_accounts
		global lastPrices
		global buyAllowed
		global dataHistoDict
		global spreadCurrencies
		global changeLastPurchase
		global changeLastPrice
		global dataTrans
		global shortAverage
		global mediumAverage
		global longAverage
		global ordersCheck
		market = self.name
		# Get lock to synchronize threads
		threadLock.acquire()

		# Do price checking for this market
		try:
			ticker = c.get_product_ticker(product_id=market)
		except:
			# Free lock to release next thread
			print('Error when getting ticker of: ' + market)
			lastPrices.pop(market, 0)
			threadLock.release()
			return
		if 'price' in ticker:
			if ticker['price'] is not None:
				lastPrices[market] = float(ticker['price'])
		else:
			print('Ticker does not contain price: ' + market)
			lastPrices.pop(market, 0)
			threadLock.release()
			return

		# Check if market is in last prices
		if not market in lastPrices:
			print('Market ' + market + ' not in lastPrices')
			# Free lock to release next thread
			threadLock.release()
			return
		
		if not market in buyAllowed:
			print('New market buyAllowed: ' + market)
			buyAllowed[market] = True

		if not market in dataTrans:
			print('No transactions recorded before:' + market)
			dataTrans[market] = np.array([np.array([currentTime]), np.array([lastPrices[market]]), np.array(['s'])], dtype=object)
		elif len(dataTrans[market][0]) == 0:
			print('No transactions recorded before:' + market)
			dataTrans[market] = np.array([np.array([currentTime]), np.array([lastPrices[market]]), np.array(['s'])], dtype=object)
		
		if not market in dataHistoDict:
			print('New market in dataHist: ' + market)
			dataHistoDict[market] = np.array([np.array([currentTime]), np.array([lastPrices[market]])], dtype=object)
		else:
			dataHistoDict[market] = [np.append(dataHistoDict[market][0], currentTime), np.append(dataHistoDict[market][1], lastPrices[market])]
		if len(dataHistoDict[market][1]) > 500:
			dataHistoDict[market] = [dataHistoDict[market][0][-500:], dataHistoDict[market][1][-500:]]
		shortAverage[market] = common.calculate_ema(dataHistoDict[market][1], shortPeriod)
		mediumAverage[market] = common.calculate_ema(dataHistoDict[market][1], mediumPeriod)
		longAverage[market] = common.calculate_ema(dataHistoDict[market][1], longPeriod)

		# Get change since last purchase
		changeLastPurchase[market] = ((lastPrices[market] - dataTrans[market][1][0]) / dataTrans[market][1][0]) * 100.0

		#Get change since last 10 prices
		if len(dataHistoDict[market][1]) > 10:
			previousPrice = dataHistoDict[market][1][-10]
		else:
			previousPrice = dataHistoDict[market][1][-1]
		changeLastPrice[market] = ((lastPrices[market] - previousPrice) / previousPrice) * 100.0

		# Calculate spread
		try:
			sp = c.get_product_order_book(market)
		except:
			print('Error when getting orderbook')
			threadLock.release()
			return
		if not 'bids' in sp or not 'asks' in sp:
			threadLock.release()
			return
		spreadCurrencies[market] = (abs(float(sp['bids'][0][0]) - float(sp['asks'][0][0])) / lastPrices[market]) * 100

		# check that is not a low fluidity pair
		if spreadCurrencies[market] >= 0.5:
			threadLock.release()
			return

		# check that there is continuous data
		if len(dataHistoDict[market][0]) >= longPeriod+1:
			if dataHistoDict[market][0][-1] - dataHistoDict[market][0][-(longPeriod+1)] > longPeriod+1:
				print('Insufficient data to trade: ' + market)
				threadLock.release()
				return
		else:
			print('Insufficient data to trade: ' + market)
			threadLock.release()
			return

		# If market goes up sell
		if mediumAverage[market][-1] > shortAverage[market][-1] and mediumAverage[market][-2] <= shortAverage[market][-2] and not buyAllowed[market] and mediumAverage[market][-1] >= longAverage[market][-1]: # and changeLastPurchase[market] > 3.0
			baseAsset = euroPairs[indexMarket[market]]['base_currency']

			if not baseAsset in indexCurrency:
				print('Currency not available: ' + market)
				threadLock.release()
				return

			available = float(my_accounts[indexCurrency[baseAsset]]['available'])

			if available != 0.0:
				# Create sell market order
				print('Selling: ' + market + ' ____ Amount: ' + str(available))
				try:
					new_sell_market_order = auth_client.place_market_order(product_id=market, side='sell', size=np.format_float_positional(available))
					print(new_sell_market_order['status'])
					if new_sell_market_order['status'] == 'done':
						dataTrans[market] = [np.append(dataTrans[market][0], currentTime), np.append(dataTrans[market][1], lastPrices[market]), np.append(dataTrans[market][2], 's')]
						buyAllowed[market] = True
						# Add successful tradde
						transactionsDone += 1
					else:
						ordersCheck.append(new_sell_market_order['id'])
				except Exception as inst:
					# Free lock to release next thread
					print(inst)
					threadLock.release()
					return

			else:
				# Unsuccessful trade
				transactionsAttempted += 1

		# # If market goes down buy
		# if longAverage[market][-1] < shortAverage[market][-1] and longAverage[market][-2] > shortAverage[market][-2] and longAverage[market][-1] >= mediumAverage[market][-1] and buyAllowed[market]:
		# 	quoteAsset = euroPairs[indexMarket[market]]['quote_currency']

		# 	if not quoteAsset in indexCurrency:
		# 		print('Currency not available: ' + market)
		# 		threadLock.release()
		# 		return

		# 	available = float(my_accounts[indexCurrency[quoteAsset]]['available'])

		# 	if available > 5.0 and available != 0.0:
		# 		# Create buy market order
		# 		print('Buying: ' + market + ' ____ Amount: ' + str(5.00))
		# 		try:
		# 			new_buy_market_order = auth_client.place_market_order(product_id=market, side='buy', funds='5.00')
		# 			print(new_buy_market_order['status'])
		# 			if new_buy_market_order['status'] == 'done':
		# 				dataTrans[market] = [np.append(dataTrans[market][0], currentTime), np.append(dataTrans[market][1], lastPrices[market]), np.append(dataTrans[market][2], 'b')]
		# 				buyAllowed[market] = False
		# 				# Add successful trade
		# 				transactionsDone += 1
		# 			else:
		# 				ordersCheck.append(new_buy_market_order['id'])
		# 		except Exception as inst:
		# 			# Free lock to release next thread
		# 			print(inst)
		# 			threadLock.release()
		# 			return

		# 	else:
		# 		# Unsuccessful trade
		# 		transactionsAttempted += 1

		# Free lock to release next thread
		threadLock.release()

def trade(t, markets, changePurchase, holding, changePrice, spread, transactions, ini=False):
	global currentTime
	currentTime = t
	global euroPairs
	global transactionsDone
	global transactionsCancelled
	global my_accounts
	global indexCurrency
	global lastPrices
	global ordersCheck
	print('#################################################################################')
	print('Checking prices etc.')
	l = len(euroPairs)
	common.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 25)

	ordersCheck = []

	try:
		# Get balance for all currencies
		my_accounts = auth_client.get_accounts()
		# Get indices for accounts
		indexCurrency = dict()
		for i in range(len(my_accounts)):
			indexCurrency[my_accounts[i]['currency']] = i
	except:
		print('Error when getting data from server')
		return

	if ini:
		getTable(markets, changePurchase, holding, changePrice, spread)
		return

	threads = []

	# Initialize threads
	for i in range(len(euroPairs)):
		market = euroPairs[i]['id']
		threads.append(coinThread(i, market))

	for t in threads:
		t.start()

	i = 0
	for t in threads:
		progreso = common.printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=25)
		i+=1
		try:
			t.join()
		except Exception as inst:
			print('Error in thread')
			print(inst)
			continue
		transactions.clear()
		# Print information
		transactions.append('Transactions done: ' + str(transactionsDone) + '\nTransactions attempted: ' + str(
			transactionsAttempted) + '\nTransactions cancelled: ' + str(transactionsCancelled)
			+ '\n' + progreso)
	getTable(markets, changePurchase, holding, changePrice, spread)

	print('Waiting for transactions to end')
	time.sleep(5)

	for orderID in ordersCheck:
		my_exchange_order = auth_client.get_order(orderID)
		check = my_exchange_order['product_id']
		print(my_exchange_order['status'])
		if my_exchange_order['status'] == 'done':
			if my_exchange_order['side'] == 'buy':
				buyAllowed[my_exchange_order['product_id']] = False
				dataTrans[check] = [np.append(dataTrans[check][0], currentTime), np.append(dataTrans[check][1], lastPrices[check]), np.append(dataTrans[check][2], 'b')]
			else:
				buyAllowed[my_exchange_order['product_id']] = True
				dataTrans[check] = [np.append(dataTrans[check][0], currentTime), np.append(dataTrans[check][1], lastPrices[check]), np.append(dataTrans[check][2], 's')]
			# Add successful tradde
			transactionsDone += 1
			print('transaction finished')
		else:
			print('transaction being cancelled')
			transactionsCancelled += 1
			cancelled_order = auth_client.cancel_order(my_exchange_order['product_id'])

	saveData()
