from HMM import  HMM
import  numpy as np
if __name__=="__main__":
	trainsition_probability =[[0.5, 0.2,0.3], [0.3,0.5,0.2 ],[0.2,0.3,0.5]]# [[0.7, 0.3], [0.4, 0.6]]
	emission_probability = np.array([[0.5, 0.5], [0.4, 0.6],[0.7,0.3]])#np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
	#pi = [0.6, 0.4]
	pi=[0.2,0.4,0.4]
	# 然后下面先得到前向算法,在A,B,pi参数已知的前提下，求出特定观察序列的概率是多少?
	obs_seq = [0,1,0]
	hmm=HMM(trainsition_probability,emission_probability,pi,obs_seq,);
	alph=hmm.forward()
	beta=hmm.Backward()
	#print(alph.transpose(),beta.transpose(),);exit()
	# 下面是得到后向算法的结果
	res_backword = 0
	res_backward = np.sum(pi * emission_probability[:,obs_seq[0]]* beta[0,:])  # 一定要乘以发射的那一列
	print("res_backward = {}".format(res_backward))
	print("res_forward = {}".format(np.sum(alph[len(obs_seq)-1,:])))

	delta, paths=hmm.viterbi()
	print(delta,paths)
