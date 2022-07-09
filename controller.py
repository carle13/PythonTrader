import requests
import _ssl
import ssl
from io import StringIO  # Python3 use: from io import StringIO
import sys
# Direct output to terminal tab
sys.stdout = sys.stderr = mystdout = StringIO()

from matplotlib.pyplot import style
from numpy import indices
from CoinbaseTrade import Coin
from NiceHashTrade import Nice
from window import Ui_MainWindow
from threading import Thread
import sys
from time import time, sleep
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QEvent, Qt
from PyQt5 import QtGui
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
import random

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

marketsNice = []
changePurchaseNice = []
holdingNice = []
changePriceNice = []
spreadNice = []
transactionsNice = []

marketsCoin = []
changePurchaseCoin = []
holdingCoin = []
changePriceCoin = []
spreadCoin = []
transactionsCoin = []

class niceChecking(QObject):
	progress = pyqtSignal()
	def run(self):
		timeNow = (time() // checkingTime) + 1
		try:
			Nice.trade(timeNow, marketsNice, changePurchaseNice, holdingNice, changePriceNice, spreadNice, transactionsNice, True)
		except Exception as inst:
			print('Error when initializing tables and graphs')
			print(inst)
		self.progress.emit()
		while(True):
			sleep(0.1)
			if time() // checkingTime >= timeNow:
				print("Check NiceHash")
				try:
					Nice.trade(timeNow, marketsNice, changePurchaseNice, holdingNice, changePriceNice, spreadNice, transactionsNice)
				except Exception as inst:
					print('Error NiceHash')
					print(inst)
				timeNow = (time() // checkingTime) + 1
				self.progress.emit()

class coinChecking(QObject):
	progress = pyqtSignal()
	def run(self):
		timeNow = (time() // checkingTime) + 1
		try:
			Coin.trade(timeNow, marketsCoin, changePurchaseCoin, holdingCoin, changePriceCoin, spreadCoin, transactionsCoin, True)
		except Exception as inst:
			print('Error when initializing tables and graphs')
			print(inst)
		self.progress.emit()
		while(True):
			sleep(0.1)
			if time() // checkingTime >= timeNow:
				print("Check Coinbase")
				try:
					Coin.trade(timeNow, marketsCoin, changePurchaseCoin, holdingCoin, changePriceCoin, spreadCoin, transactionsCoin)
				except Exception as inst:
					print('Error Coinbase')
					print(inst)
				timeNow = (time() // checkingTime) + 1
				self.progress.emit()

class updateUI(QObject):
	progress = pyqtSignal()
	def run(self):
		while(True):
			sleep(0.1)
			self.progress.emit()

checkingTime = 900  # Check prices and trade if possible every minute

print("modules were imported")

class controllerWindow(qtw.QMainWindow):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		style = self.style()

		self.tray_icon = qtw.QSystemTrayIcon()
		self.tray_icon.setIcon(QtGui.QIcon('btc-logo.png'))
		self.setWindowIcon(QtGui.QIcon('btc-logo.png'))

		# Restore the window when the tray icon is double clicked.
		self.tray_icon.activated.connect(self.restore_window)

		self.ui.TableData.cellClicked.connect(self.marketClickedNice)
		self.ui.TableData_2.cellClicked.connect(self.marketClickedCoin)

		# Set plots for NiceHash
		self.niceMarket = plt.figure()
		self.niceBase = plt.figure()
		self.ui.graphMarket = FigureCanvas(self.niceMarket)
		self.ui.graphBase = FigureCanvas(self.niceBase)
		self.toolbar = NavigationToolbar(self.ui.graphMarket, self.ui.graphMarket)
		self.toolbar2 = NavigationToolbar(self.ui.graphBase, self.ui.graphBase)

		self.ui.ImagesSplitter.addWidget(self.ui.graphMarket)
		self.ui.ImagesSplitter.addWidget(self.ui.graphBase)

		self.baseNice = 'ETHBTC'

		self.niceMarket.clear()
		# create an axis
		ax = self.niceMarket.add_subplot(111)
		# plot data
		Nice.plotBase(self.baseNice, ax)
		# refresh canvas
		self.ui.graphMarket.draw()

		# Set plots for Coinbase
		self.coinMarket = plt.figure()
		self.coinBase = plt.figure()
		self.ui.graphMarket_2 = FigureCanvas(self.coinMarket)
		self.ui.graphBase_2 = FigureCanvas(self.coinBase)
		self.toolbar_2 = NavigationToolbar(self.ui.graphMarket_2, self.ui.graphMarket_2)
		self.toolbar2_2 = NavigationToolbar(self.ui.graphBase_2, self.ui.graphBase_2)

		self.ui.ImagesSplitter_2.addWidget(self.ui.graphMarket_2)
		self.ui.ImagesSplitter_2.addWidget(self.ui.graphBase_2)

		self.baseCoin = 'BTC-EUR'

		self.coinMarket.clear()
		# create an axis
		ax = self.coinMarket.add_subplot(111)
		# plot data
		Coin.plotBase(self.baseCoin, ax)
		# refresh canvas
		self.ui.graphMarket_2.draw()

		# Start threads
		self.threadNice = QThread()
		self.niceChecker = niceChecking()
		self.niceChecker.moveToThread(self.threadNice)
		self.threadNice.started.connect(self.niceChecker.run)
		self.threadNice.finished.connect(self.threadNice.deleteLater)
		self.niceChecker.progress.connect(self.niceTable)
		self.threadNice.start()

		self.threadCoin = QThread()
		self.coinChecker = coinChecking()
		self.coinChecker.moveToThread(self.threadCoin)
		self.threadCoin.started.connect(self.coinChecker.run)
		self.threadCoin.finished.connect(self.threadCoin.deleteLater)
		self.coinChecker.progress.connect(self.coinTable)
		self.threadCoin.start()

		self.thread2 = QThread()
		self.updateUi = updateUI()
		self.updateUi.moveToThread(self.thread2)
		self.thread2.started.connect(self.updateUi.run)
		self.thread2.finished.connect(self.thread2.deleteLater)
		self.updateUi.progress.connect(self.updateImage)
		self.thread2.start()
	
	def updatePlotsNice(self):
		self.niceMarket.clear()
		# create an axis
		ax = self.niceMarket.add_subplot(111)
		# plot data
		Nice.plotBase(self.baseNice, ax)
		# refresh canvas
		self.ui.graphMarket.draw()

	def updatePlotsCoin(self):
		self.coinMarket.clear()
		# create an axis
		ax = self.coinMarket.add_subplot(111)
		# plot data
		Coin.plotBase(self.baseCoin, ax)
		# refresh canvas
		self.ui.graphMarket_2.draw()

	def changeEvent(self, event):
		if event.type() == QEvent.WindowStateChange:
			if event.oldState() and Qt.WindowMinimized:
				self.hide()
				self.tray_icon.show()
			elif event.oldState() == Qt.WindowNoState or self.windowState() == Qt.WindowMaximized:
				print("WindowMaximized")
	
	def restore_window(self, reason):
		if reason == qtw.QSystemTrayIcon.DoubleClick:
			# self.showNormal will restore the window even if it was
			# minimized.
			self.show()
			self.showMaximized()

	def marketClickedNice(self):
		# Read selected row
		self.indices = self.ui.TableData.selectionModel().selectedRows()
		for index in sorted(self.indices):
			print('Row %d is selected' % index.row())
			print(self.ui.TableData.item(index.row(), 0).text())
			self.baseNice = self.ui.TableData.item(index.row(), 0).text()
			self.updatePlotsNice()
	
	def marketClickedCoin(self):
		# Read selected row
		self.indices = self.ui.TableData_2.selectionModel().selectedRows()
		for index in sorted(self.indices):
			print('Row %d is selected' % index.row())
			print(self.ui.TableData_2.item(index.row(), 0).text())
			self.baseCoin = self.ui.TableData_2.item(index.row(), 0).text()
			self.updatePlotsCoin()

	
	def updateImage(self):
		# Update graphs NiceHash
		# self.graph1 = QtGui.QPixmap(pictureNice1)
		# self.graph1Scaled = self.graph1.scaled(
		# 	self.ui.graphMarket.width(), self.ui.graphMarket.height(), qtc.Qt.AspectRatioMode.KeepAspectRatio)

		# self.graph2 = QtGui.QPixmap(pictureNice2)
		# self.graph2Scaled = self.graph2.scaled(self.ui.graphBase.width(
		# ), self.ui.graphBase.height(), qtc.Qt.AspectRatioMode.KeepAspectRatio)

		# self.ui.graphMarket.setPixmap(self.graph1Scaled)
		# self.ui.graphBase.setPixmap(self.graph2Scaled)

		# # Update graphs Coinbase
		# self.graph1_2 = QtGui.QPixmap(pictureCoin1)
		# self.graph1Scaled_2 = self.graph1_2.scaled(
		# 	self.ui.graphMarket_2.width(), self.ui.graphMarket_2.height(), qtc.Qt.AspectRatioMode.KeepAspectRatio)

		# self.graph2_2 = QtGui.QPixmap(pictureCoin2)
		# self.graph2Scaled_2 = self.graph2_2.scaled(self.ui.graphBase_2.width(
		# ), self.ui.graphBase_2.height(), qtc.Qt.AspectRatioMode.KeepAspectRatio)

		# self.ui.graphMarket_2.setPixmap(self.graph1Scaled_2)
		# self.ui.graphBase_2.setPixmap(self.graph2Scaled_2)

		#Update transactions text
		for i in range(len(transactionsNice)):
			self.ui.transactions.setText(transactionsNice[i])
		for i in range(len(transactionsCoin)):
			self.ui.transactions_2.setText(transactionsCoin[i])
		
		#Update terminal text
		self.ui.terminalText.setText(mystdout.getvalue()[-1000:])

	def niceTable(self):
		# Update plots NiceHash
		self.updatePlotsNice()

		# Update table NiceHash
		self.ui.TableData.clearContents()
		self.ui.TableData.setRowCount(0)
		self.ui.TableData.setRowCount(len(marketsNice))
		self.ui.TableData.setSortingEnabled(False)
		for i in range(len(marketsNice)):
			self.ui.TableData.setItem(i, 0, qtw.QTableWidgetItem(marketsNice[i]))

			self.ui.TableData.setItem(
							i, 1, qtw.QTableWidgetItem(str(holdingNice[i])))
			if holdingNice[i]:
				self.ui.TableData.item(i, 1).setBackground(QtGui.QColorConstants.Yellow)

			self.ui.TableData.setItem(i, 2, qtw.QTableWidgetItem(
				str("{:.2f}".format(changePurchaseNice[i]))+'%'))
			if(changePurchaseNice[i] > 5):
				self.ui.TableData.item(i, 2).setBackground(QtGui.QColorConstants.Green)
			elif (changePurchaseNice[i] < -4):
				self.ui.TableData.item(i, 2).setBackground(QtGui.QColorConstants.Red)

			self.ui.TableData.setItem(i, 3, qtw.QTableWidgetItem(
				str("{:.2f}".format(changePriceNice[i]))+'%'))
			if(changePriceNice[i] > 0):
				self.ui.TableData.item(i, 3).setBackground(QtGui.QColorConstants.Green)
			elif (changePriceNice[i] < 0):
				self.ui.TableData.item(i, 3).setBackground(QtGui.QColorConstants.Red)
			
			self.ui.TableData.setItem(i, 4, qtw.QTableWidgetItem(
				str("{:.2f}".format(spreadNice[i]))+'%'))
			if(spreadNice[i] <= 0.5):
				self.ui.TableData.item(i, 4).setBackground(QtGui.QColorConstants.Green)
			else:
				self.ui.TableData.item(i, 4).setBackground(QtGui.QColorConstants.Red)
		self.ui.TableData.setSortingEnabled(True)

	def coinTable(self):
		# Update plots Coinbase
		self.updatePlotsCoin()

		# Update table Coinbase
		self.ui.TableData_2.clearContents()
		self.ui.TableData_2.setRowCount(0)
		self.ui.TableData_2.setRowCount(len(marketsCoin))
		self.ui.TableData_2.setSortingEnabled(False)
		for i in range(len(marketsCoin)):
			self.ui.TableData_2.setItem(i, 0, qtw.QTableWidgetItem(marketsCoin[i]))

			self.ui.TableData_2.setItem(
                            i, 1, qtw.QTableWidgetItem(str(holdingCoin[i])))
			if holdingCoin[i]:
				self.ui.TableData_2.item(i, 1).setBackground(QtGui.QColorConstants.Yellow)

			self.ui.TableData_2.setItem(i, 2, qtw.QTableWidgetItem(
				str("{:.2f}".format(changePurchaseCoin[i]))+'%'))
			if(changePurchaseCoin[i] > 5):
				self.ui.TableData_2.item(i, 2).setBackground(QtGui.QColorConstants.Green)
			elif (changePurchaseCoin[i] < -4):
				self.ui.TableData_2.item(i, 2).setBackground(QtGui.QColorConstants.Red)

			self.ui.TableData_2.setItem(i, 3, qtw.QTableWidgetItem(
				str("{:.2f}".format(changePriceCoin[i]))+'%'))
			if(changePriceCoin[i] > 0):
				self.ui.TableData_2.item(i, 3).setBackground(QtGui.QColorConstants.Green)
			elif (changePriceCoin[i] < 0):
				self.ui.TableData_2.item(i, 3).setBackground(QtGui.QColorConstants.Red)

			self.ui.TableData_2.setItem(i, 4, qtw.QTableWidgetItem(
				str("{:.2f}".format(spreadCoin[i]))+'%'))
			if(spreadCoin[i] <= 0.5):
				self.ui.TableData_2.item(i, 4).setBackground(QtGui.QColorConstants.Green)
			else:
				self.ui.TableData_2.item(i, 4).setBackground(QtGui.QColorConstants.Red)
		self.ui.TableData_2.setSortingEnabled(True)
		

app = qtw.QApplication([])
widget = controllerWindow()
widget.showMaximized()
widget.updateImage()

sys.exit(app.exec_())
