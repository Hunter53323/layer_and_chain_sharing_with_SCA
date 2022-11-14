import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from simple_deployment import simple_deployment_main, random_deployment, layer_match_deployment, k8s_deployment
from matrix import matrix
from utils import decoder, create_microservice
import psutil
import os

def show_info():
    """
    显示当前进程的内存占用
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    info = py.memory_full_info()
    memory = info.uss/1024/1024
    return memory

def get_start_value(theta, para_dict):
    """
    从不同的函数里面得到求解的初始值
    """
    deployment = simple_deployment_main(theta,para_dict)
    # deployment = random_deployment()
    x_start, d_start = deployment.deployment_encoder()
    
    return x_start, d_start

def Gurobi(opt = 0, K = 9, N = 9, T_max = 1, T_min = 0, R_max = 1, R_min = 0, theta = 0.5, get_max_min:bool = False, start_value:bool = False, SCA:bool = False, output:bool = True, Model_output = True, only_model = False):
    """
    支持任意情况进行求解，全部封装到同一个函数里面
    支持SCA方法和原始方法、支持有初始解和无初始解、支持帕累托最优和原本
    opt = 0: 双目标优化;opt = 1:单独优化通讯开销(R);opt = 2:单独优化下载延迟(T)
    input_matrix用于判断有没有现成的输入矩阵,如果有输入矩阵的话那么就可以直接使用,不用连接matlab
    """
    memory1 = show_info()

    SCA_num = 1
    M,W,Q,b,Y,S,G,C_S,C_C,H,VG,x_dim,d_dim,P_SCA,N_SCA, para_dict = matrix(K,N,SCA)
    if only_model:
        return para_dict
    if SCA:
        x_last = np.zeros(int(x_dim))
        epsilon = 100
        SCA_num = 20

    # 初始化Gurobi
    model = gp.Model()
    # 是否使用SCA,以及是否需要输出
    if SCA == True or output == False or Model_output == False:
        model.setParam('OutputFlag', 0)
    model.setParam('nonconvex', 2)
    x = model.addVars(int(x_dim), vtype=GRB.BINARY, name='x')
    d = model.addVars(int(d_dim), vtype=GRB.BINARY, name='d')
    # x = model.addVars(int(x_dim), vtype=GRB.CONTINUOUS, name='x')
    # d = model.addVars(int(d_dim), vtype=GRB.CONTINUOUS, name='d')

    # 添加约束
    model.addConstrs(gp.quicksum(Q[i][j]*x[j] for j in range(x_dim)) == b[i][0] for i in range(b.size))
    model.addConstrs(gp.quicksum(H[i][j]*x[j] for j in range(x_dim)) == VG[i][0] for i in range(len(VG)))
    model.addConstrs(gp.quicksum(Y[i][j]*x[j] for j in range(x_dim)) >= d[i] for i in range(d_dim))
    model.addConstrs(gp.quicksum(Y[i][j]*x[j]/10 for j in range(x_dim)) <= d[i] for i in range(d_dim))
    model.addConstrs(gp.quicksum(S[i][j]*d[j] for j in range(d_dim)) <= C_S[i] for i in range(C_S.size))
    model.addConstrs(gp.quicksum(G[i][j]*x[j] for j in range(x_dim)) <= C_C[i] for i in range(C_C.size))

    start = time.perf_counter()

    if get_max_min == True:
        #minmax的第一步
        obj = [x[i]*x[j]*W[i][j] for i in range(x_dim) for j in range(x_dim)]
        model.setObjective(gp.quicksum(obj), GRB.MINIMIZE)
        print("开始求解第一个最大最小值")
        model.optimize()
        print("求解完成")

        T_max = sum([d[i].x*float(M[0][i]) for i in range(d_dim)])
        R_min = sum([x[i].x*x[j].x*W[i][j] for i in range(x_dim) for j in range(x_dim)])

        #minmax的第二步
        obj = [d[i]*float(M[0][i]) for i in range(d_dim)]
        model.setObjective(gp.quicksum(obj), GRB.MINIMIZE)
        print("开始求解第二个最大最小值")
        model.optimize()
        print("求解完成")

        T_min = sum([d[i].x*float(M[0][i]) for i in range(d_dim)])
        R_max = sum([x[i].x*x[j].x*W[i][j] for i in range(x_dim) for j in range(x_dim)])

    Tconst1 = theta/(T_max - T_min)
    Tconst2 = float(theta*T_min)/(T_max - T_min)
    Rconst1 = (1-theta)/(R_max - R_min)
    Rconst2 = float((1-theta)*R_min)/(R_max - R_min)

    # 是否有求解初值
    if start_value == True and opt == 0:
        x_start, d_start = get_start_value(theta, para_dict)
        for i in range(int(x_dim)):
            x[i].start = int(x_start[i])
        for i in range(int(d_dim)):
            d[i].start = int(d_start[i])
        
    for i in range(SCA_num):
        obj = []
        if opt == 0 or opt == 1:
            for i in range(int(x_dim)):
                for j in range(int(x_dim)):
                    if SCA == False:
                        obj.append(Rconst1 * x[i]*x[j]*W[i][j])
                    else:
                        obj.append(Rconst1 * 0.5 * x[i]*x[j]*P_SCA[i][j])
                        obj.append(Rconst1 * (-1) * x_last[i]*x[j]*N_SCA[i][j])
        if opt == 0 or opt == 2:
            for i in range(int(d_dim)):
                    obj.append(Tconst1 * d[i]*float(M[0][i]))
        model.setObjective(gp.quicksum(obj), GRB.MINIMIZE)
        print("开始优化")
        model.optimize()

        # if model.status == GRB.Status.INFEASIBLE:
        #     print('Optimization was stopped with status %d' % model.status)
        #     # do IIS, find infeasible constraints
        #     model.computeIIS()
        #     model.write("model1.ilp")
        #     for c in model.getConstrs():
        #         if c.IISConstr:
        #             print('%s' % c.constrName)

        x_res = []
        for i in range(int(x_dim)):
            x_res.append(x[i].x)
        if SCA == True:
            epsilon = sum(np.abs(np.array(x_res) - x_last))
            x_last = x_res
            print("epsilon:",epsilon)
            if epsilon < 1:
                break
    memory3 = show_info()
    end = time.perf_counter()

    d_res = []
    for i in range(int(d_dim)):
        d_res.append(d[i].x)

    # 计算实际的通讯开销与拉取延迟
    layer = 0
    for i in range(d_dim):
        layer += d_res[i]*float(M[0][i])
    
    chain = 0
    for i in range(x_dim):
        for j in range(x_dim):
            chain += x_res[i]*x_res[j]*W[i][j]

    deployment = 0
    if output == True:
        deployment,x_fix,d_ori,d_fix = decoder(x_res,K,N,para_dict)
        L = para_dict['L']
        S_l = np.array(para_dict['S_l']).reshape(int(L))
        
        storage = sum([S_l[l]*d_res[n*int(L)+l] for l in range(int(L)) for n in range(int(N))])
        print(deployment)
        print("applications:",para_dict["K"],"servers:",para_dict["N"],"microservices:",sum(para_dict["A"]))
        print("storage:",storage,"layer:",layer,"chain:",chain)
        print("T_max:",T_max,"T_min:",T_min,"R_max:",R_max,"R_min:",R_min)
        print("T_const1:",Tconst1,"T_const2:",Tconst2,"R_const1:",Rconst1,"R_const2:",Rconst2)
        print("objective value:",layer*Tconst1+chain*Rconst1-Tconst2-Rconst2)
        print("layer_objectivevalue:",T_min*Tconst1+R_max*Rconst1-Tconst2-Rconst2)
        print("chain_objectivevalue:",T_max*Tconst1+R_min*Rconst1-Tconst2-Rconst2)
    print("time:",end-start,"s"," memory:",[memory1,memory3])
    TR = [Tconst1,Tconst2,Rconst1,Rconst2]
    para_dict['TR'] = TR
    para_dict['layer'] = layer
    para_dict['chain'] = chain

    # return x_res,d_res,model.objVal, layer, chain, deployment
    # return x_res,d_res,model.objVal, layer, chain, layer*Tconst1+chain*Rconst1
    return para_dict

if __name__ == "__main__":
    # layer_list = []
    # chain_list = []
    # for i in range(10):
    #     cus_theta = 0.999
    #     para_dict = Gurobi(opt=0, K = 20, N = 15, get_max_min=True, SCA=False, start_value=False, theta = cus_theta)
    #     print("------------------------")
    #     deployment = simple_deployment_main(theta= 1-cus_theta, para_dict = para_dict)
    #     # print("总存储:",deployment.calculate_storage_total())
    #     # print("总通讯:",deployment.calculate_communication_total())
    #     layer_list.append(para_dict['layer']/deployment.calculate_storage_total())
    #     chain_list.append(para_dict['chain']/deployment.calculate_communication_total())
    # print("layer:",sum(layer_list)*100/len(layer_list))
    # print("chain:",sum(chain_list)/len(chain_list))
    # print("-------greedy----------")
    # # print("simple deployment:",deployment.get_deployment())
    # print("storage:",deployment.calculate_storage_occupation(),"download time:",deployment.calculate_download_time(),"communication:",deployment.calculate_communication())
    # objective = para_dict['TR'][0]*deployment.calculate_download_time()+para_dict['TR'][2]*deployment.calculate_communication()-para_dict['TR'][1]-para_dict['TR'][3]
    # print("objective value:",objective)
    # print("-------layer match---------")
    # deployment = layer_match_deployment(para_dict = para_dict)
    # # print("simple deployment:",deployment.get_deployment())
    # print("storage:",deployment.calculate_storage_occupation(),"download time:",deployment.calculate_download_time(),"communication:",deployment.calculate_communication())
    # objective = para_dict['TR'][0]*deployment.calculate_download_time()+para_dict['TR'][2]*deployment.calculate_communication()-para_dict['TR'][1]-para_dict['TR'][3]
    # print("objective value:",objective)
    # print("--------k8s-----------")
    # deployment = k8s_deployment(para_dict = para_dict)
    # # print("simple deployment:",deployment.get_deployment())
    # print("storage:",deployment.calculate_storage_occupation(),"download time:",deployment.calculate_download_time(),"communication:",deployment.calculate_communication())
    # objective = para_dict['TR'][0]*deployment.calculate_download_time()+para_dict['TR'][2]*deployment.calculate_communication()-para_dict['TR'][1]-para_dict['TR'][3]
    # print("objective value:",objective)
    # print("------------------------")
    start = time.perf_counter()
    cus_theta = 0.5
    para_dict = Gurobi(opt=0, K = 10, N = 10, get_max_min=True, SCA=False, start_value=False, theta = cus_theta, only_model=True)
    print("------------------------")
    deployment = simple_deployment_main(theta= 1-cus_theta, para_dict = para_dict)
    # print("总存储:",deployment.calculate_storage_total())
    # print("总通讯:",deployment.calculate_communication_total())
    end = time.perf_counter()
    print("time:",end-start,"s")
    print(show_info())
