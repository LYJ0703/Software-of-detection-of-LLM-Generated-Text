# LLM生成文本检测器
（本应用使用的生成文本检测方法需要借助大语言模型，需要自行导入）

1.前期网页app是用gradio库编写的，gradio中有一些连接外网的操作，而我们的实验环境是在集群上，连接外网的功能开启很麻烦，所因此使用脚本作为中间连接，运行webapp_sh.py后，程序会并行开启几个脚本针对不同的任务分配gpu进行计算，把所有结果写入文件中后，再由webapp_sh.py程序呈现。

2.webapp.py是demo演示，对应无需集群的应用场景，使用前需要将模型下载到本地，模型可以自行选择并放入model目录中，同时修改代码中的路径。

3.目前我们的网页基于Vue实现，并用flask进行前端和服务器端的交互。网页相关文件位于vite文件夹中，使用时需运行run_app.sh和前端进行交互。

4.文件说明：

	single_roberta.py:计算预训练语言模型的softmax
	 
	ppl.py：计算ppl
	 
	Bscore.py：计算DNA-GPT中的Bscore
	 
	single_three.py：接收上面三个文件的输出，使用我们的多特征方法进行计算
	
	app.py：运行时启动的flask交互代码
	 
	webapp_sh.py：在集群环境下，运行gradio库编写的网页并控制启动脚本
	 
	webapp.py：demo演示启动程序
	 
	utiles.py：一些与模型相关的操作

4.目录结构说明：

	dataset：使用到的数据集示例展示
	 
	eg：各种计算结果示例展示
	 
	eval：与评估相关的代码
	 
	model：存放实验需要的模型（需要自行导入）
	 
	pt：训练参数。因为github上有上传文件大小限制，请访问链接https://drive.google.com/drive/folders/17xg8SP94GbjEbVn_G3HxUJva34zIts7m?usp=sharing下载模型参数。
	 
	sh：集群应用下的脚本
	 
	train：与训练相关的代码
	
	vite：Vue前端代码
	
	app：flask交互代码
