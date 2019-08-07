import numpy as np
import  random

class HMM(object):
	def __init__(self,trainsition,emission,pi,obs_seq,):
		"""
		    :param trainsition_probability:trainsition_probability是状态转移矩阵
		    :param emission_probability: emission_probability是发射矩阵
		    :param pi: pi是初始状态概率
		    :param obs_seq: obs_seq是观察状态序列
		    :return: 返回结果
		    """
		self.pi=np.array(pi)
		self.trainsition=np.array(trainsition)
		self.emission=np.array(emission)
		self.obs_seq=np.array(obs_seq,dtype=np.int)
	#all 没问题
	def forward(self):
		trainsition_pro=np.array(self.trainsition)#s*s
		emission_pro=np.array(self.emission)#s,tag
		pi=np.array(self.pi)
		col=(trainsition_pro).shape[0]
		row=len(self.obs_seq)#t step
		alph=np.zeros((row,col))
		alph[0,:]=pi*emission_pro[:,self.obs_seq[0]]
		# np.dot 向量的內积，最后元素向加
		#print(pi,emission_pro[:,self.obs_seq[0]],np.dot(pi,emission_pro[:,self.obs_seq[0]]));exit()
		for t in range(1,row):
			for coli in range(col):
				alph[t,coli]=np.dot(alph[t-1,:],trainsition_pro[:,coli])*emission_pro[coli,self.obs_seq[t]]
		return  alph
	def Backward(self):
		trainsition_pro=np.array(self.trainsition)
		emission_pro=np.array(self.emission)
		pi=np.array(self.pi)
		row=len(self.obs_seq)
		col=trainsition_pro.shape[0]
		beta=np.zeros((row,col))
		beta[row-1,:]=1 #最后的每一个元素赋值为1
		for t in reversed(range(row-1)):
			for coli in range(col):
				beta[t,coli]=np.sum(trainsition_pro[coli,:]*emission_pro[:,self.obs_seq[t+1]]*beta[t+1,:])
		return  beta
	def viterbi(self):
		row=self.trainsition.shape[0]
		col=self.obs_seq.shape[0]
		delta=np.zeros((row,col))
		psi=np.zeros((row,col))#paths
		delta[:,0]=self.pi*self.emission[:,self.obs_seq[0]]
		psi[:,0]=0;
		for s in range(row):
			for t in range(1,col):
				delta[s,t]=np.max(delta[:,t-1]*self.trainsition[:,s])*self.emission[s,self.obs_seq[t]]
				psi[s,t]=np.argmax(delta[:,t-1]*self.trainsition[:,s])

		paths_prob=np.max(delta[:,col-1])
		paths=np.arange(col,dtype=np.int)
		paths[col-1]=np.argmax(delta[:, col-1])
		for t in reversed(range(1,col)):
			paths[t-1]=(psi[(paths[-1]),t])
		return paths_prob,paths
