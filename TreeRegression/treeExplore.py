#coding:utf8
from numpy import *
from Tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import regTrees

def reDraw(tolS, tolN):
	reDraw.f.clf()
	reDraw.a = reDraw.f.add_subplot(111)
	if chkBtnVar.get(): # 复选框是否选中
		if tolN < 2 : tolN = 2
		myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,\
									 regTrees.modelErr, (tolS, tolN))
		yHat = regTrees.createForeCast(myTree, reDraw.testDat,\
									   regTrees.modelTreeEval)
	else:
		myTree = regTrees.createTree(reDraw.rawDat,ops=(tolS, tolN))
		yHat = regTrees.createForeCast(myTree, reDraw.testDat)
	reDraw.a.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1],s=5)
	reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
	reDraw.canvas.show()

def getInputs():
	try:
		tolN = int(tolNentry.get())
	except:
		tolN = 10
		print ("enter Integer for tolN")
		tolNentry.delete(0, END)
		tolNentry.insert(0, '10')
	try:
		tolS = float(tolSentry.get())
	except:
		tolS = 1.0
		print ("enter Integer for tolS")
		tolSentry.delete(0, END)
		tolSentry.insert(0, '1.0')
	return tolS, tolN

def drawNewTree():
	tolS, tolN = getInputs()
	reDraw(tolS, tolN)

root = Tk()


reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

# Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)

#界面设计
Label(root, text="tolN").grid(row=1, column=0) # tolN标签
tolNentry = Entry(root) # 文本框
tolNentry.grid(row=1, column=1) # 文本框位置
tolNentry.insert(0,'10') # 文本框默认值为 10；
Label(root, text="tolS").grid(row=2, column=0) # tolS标签
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
# 按钮
Button(root, text="ReDraw",command=drawNewTree).grid(row=1, column=2,\
													 rowspan=3)
chkBtnVar = IntVar() # 勾选按钮的值：bool
chkBtn = Checkbutton(root, text='Model Tree', variable= chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

# 数据来源
reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),\
						max(reDraw.rawDat[:,0]), 0.01)

reDraw(1.0, 10)

root.mainloop()