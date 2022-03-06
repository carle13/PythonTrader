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

def saveData():
	global pairsBitcoin
	global lastPrices
	global currentTime
	global dataHistoDict
	global dataTrans
	# Update file data
	file = open("NiceHashTrade/buyAllowed.json", "w")
	json.dump(buyAllowed, file, sort_keys=True)
	file.close()

	# Save historical data into different csv files
	for i in range(len(pairsBitcoin)):
		market = pairsBitcoin[i]['symbol']
		if market not in lastPrices:
			# Get exchange market info
			exchange_info = public_api.get_exchange_markets_info()
			pairsBitcoin = list(exchange_info['symbols'][i] for i in range(len(exchange_info['symbols'])) if exchange_info['symbols'][i]['quoteAsset'] == 'BTC' and exchange_info['symbols'][i]['status'] == 'TRADING')
			# Get indices for market symbols
			indexMarket = dict()
			for i in range(len(pairsBitcoin)):
				indexMarket[pairsBitcoin[i]['symbol']] = i
			continue

		np.save('NiceHashTrade/HistoricalData/historical' + market +'.npy', dataHistoDict[market])

		if dataTrans[market][0][0] < dataHistoDict[market][0][0]:
			dataTrans[market] = [dataTrans[market][0][1:], dataTrans[market][1][1:], dataTrans[market][2][1:]]
		np.save('NiceHashTrade/TransactionData/transactions' + market + '.npy', dataTrans[market])


def getTable(markets, changePurchase, holding, changePrice, spread):
	markets.clear()
	changePurchase.clear()
	holding.clear()
	changePrice.clear()
	spread.clear()
	for i in range(len(pairsBitcoin)):
		key = pairsBitcoin[i]['symbol']
		markets.append(key)
		holding.append(not buyAllowed[key])
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
longPeriod = 20

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
pairsBitcoin = list(exchange_info['symbols'][i] for i in range(len(exchange_info['symbols'])) if exchange_info['symbols'][i]['quoteAsset'] == 'BTC' and exchange_info['symbols'][i]['status'] == 'TRADING')

# Get last prices
lastPrices = dict()
# lastPrices = public_api.get_exchange_lastPrices()


#########################################
# TRADING
# Read file data to see which currencies are currently on hold
try:
	file = open('NiceHashTrade/buyAllowed.json', 'r')
	buyAllowed = json.load(file)
	file.close()
except:
	buyAllowed = dict()
	for i in range(len(pairsBitcoin)):
		market = pairsBitcoin[i]['symbol']
		buyAllowed[market] = True

# Read historical prices for the currencies
dataHistoDict = dict()
shortAverage = dict()
mediumAverage = dict()
longAverage = dict()
for i in range(len(pairsBitcoin)):
	market = pairsBitcoin[i]['symbol']
	try:
		dataHistoDict[market] = np.load('NiceHashTrade/HistoricalData/historical' + market + '.npy', allow_pickle=True)
		shortAverage[market] = common.calculate_ema(dataHistoDict[market][1], shortPeriod)
		mediumAverage[market] = common.calculate_ema(dataHistoDict[market][1], mediumPeriod)
		longAverage[market] = common.calculate_ema(dataHistoDict[market][1], longPeriod)
	except:
		print('New market: ' + market)

#Read transaction history
dataTrans = dict()
for i in range(len(pairsBitcoin)):
	market = pairsBitcoin[i]['symbol']
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

# Get indices for market symbols
indexMarket = dict()
for i in range(len(pairsBitcoin)):
	indexMarket[pairsBitcoin[i]['symbol']] = i

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

		# Price checking done in the main function

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
			sp = public_api.get_exchange_orderbook(market, 1)
		except:
			print('Error when getting orderbook')
			threadLock.release()
			return
		spreadCurrencies[market] = (abs(sp['buy'][0][0] - sp['sell'][0][0]) / lastPrices[market]) * 100

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
		if mediumAverage[market][-1] > shortAverage[market][-1] and mediumAverage[market][-2] <= shortAverage[market][-2] and not buyAllowed[market] and mediumAverage[market][-1] >= longAverage[market][-1] and changeLastPurchase[market] > 1.0:
			baseAsset = pairsBitcoin[indexMarket[market]]['baseAsset']

			if not baseAsset in indexCurrency:
				print('Currency not available: ' + market)
				threadLock.release()
				return

			available = float(my_accounts['currencies'][indexCurrency[baseAsset]]['available'])
			minBase = float(pairsBitcoin[indexMarket[market]]['priMinAmount'])

			if available > minBase and available != 0.0:
				# Create sell market order
				print('Selling: ' + market + ' ____ Amount: ' + str(available))
				print('Min amount: ' + str(minBase))
				try:
					new_sell_market_order = private_api.create_exchange_sell_market_order(
						market, available)
					print(new_sell_market_order['state'])
					if new_sell_market_order['state'] == 'FULL':
						dataTrans[market] = [np.append(dataTrans[market][0], currentTime), np.append(dataTrans[market][1], lastPrices[market]), np.append(dataTrans[market][2], 's')]
						buyAllowed[market] = True
						# Add successful tradde
						transactionsDone += 1
					else:
						marketsCheck.append(market)
						print(new_sell_market_order['state'])
				except Exception as inst:
					# Free lock to release next thread
					print(inst)
					threadLock.release()
					return

			else:
				# Unsuccessful trade
				transactionsAttempted += 1

		# If market goes down buy
		if longAverage[market][-1] < shortAverage[market][-1] and longAverage[market][-2] > shortAverage[market][-2] and longAverage[market][-1] >= mediumAverage[market][-1] and buyAllowed[market]:
			quoteAsset = pairsBitcoin[indexMarket[market]]['quoteAsset']

			if not quoteAsset in indexCurrency:
				print('Currency not available: ' + market)
				threadLock.release()
				return

			available = float(my_accounts['currencies'][indexCurrency[quoteAsset]]['available'])
			minQuote = float(exchange_info['symbols'][indexMarket[market]]['secMinAmount'])

			if available > 0.0001 and available != 0.0:
				# Create buy market order
				print('Buying: ' + market + ' ____ Amount: ' + str(0.0001))
				try:
					new_buy_market_order = private_api.create_exchange_buy_market_order(market, 0.0001)
					print(new_buy_market_order['state'])
					if new_buy_market_order['state'] == 'FULL':
						dataTrans[market] = [np.append(dataTrans[market][0], currentTime), np.append(dataTrans[market][1], lastPrices[market]), np.append(dataTrans[market][2], 'b')]
						buyAllowed[market] = False
						# Add successful tradde
						transactionsDone += 1
					else:
						marketsCheck.append(market)
						print(new_buy_market_order['state'])
				except Exception as inst:
					# Free lock to release next thread
					print(inst)
					threadLock.release()
					return

			else:
				# Unsuccessful trade
				transactionsAttempted += 1

		# Free lock to release next thread
		threadLock.release()


# Get balance for all currencies
my_accounts = private_api.get_accounts()
# Get last prices
lastPrices = public_api.get_exchange_lastPrices()

def trade(t, markets, changePurchase, holding, changePrice, spread, transactions, ini=False):
	global currentTime
	currentTime = t
	global pairsBitcoin
	global transactionsDone
	global transactionsCancelled
	global my_accounts
	global lastPrices
	global marketsCheck
	print('#################################################################################')
	print('Checking prices etc.')
	l = len(pairsBitcoin)
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
	for i in range(len(pairsBitcoin)):
		market = pairsBitcoin[i]['symbol']
		threads.append(niceThread(i, market))

	for t in threads:
		t.start()

	i = 0
	for t in threads:
		progreso = common.printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=25)
		i += 1
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

	saveData()
