import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt 
# 创建微服务参数、创建拓扑、编码解码等

def decoder(x,K,N,para_dict):
    # _,_,_,_,A,L,E_kil,_,_,_,_,_,_ = create_microservice(K,N)
    A = para_dict['A']
    L = para_dict['L']
    E_kil = para_dict['E_kil']

    d_ori = np.zeros((N,L))
    A_dim = N*sum(A)
    x_fix = np.zeros((A_dim,1))

    deployment = []

    count_ms = 0
    for k in range(K):
        deployment_k = np.zeros(A[k])
        for i in range(A[k]):
            det_v = x[count_ms*N:(count_ms+1)*N]
            n = np.argmax(det_v)
            deployment_k[i] = n
            x_fix[count_ms*N+n-1]=1
            d_ori[n] += E_kil[k][i]
            count_ms += 1
        deployment.append(deployment_k)

    d_fix = np.zeros((N,L))
    d_fix = np.where(d_ori>0.9,1,d_fix)

    return deployment,x_fix,d_ori,d_fix

# 生成微服务与层
def create_microservice(K = 10, N = 10):
    # K为应用程序数量
    # N为边缘服务器数量

    # C_S = 1000*np.array([16]*N).transpose()
    # C_C = np.array([1.6*4]*N).transpose()
    C_S = 1000*np.array([8]*N).transpose()
    C_C = np.array([1.6*2]*N).transpose()
    b_cloud = np.array([100]*N)

    #A矩阵代表每个应用程序k的微服务数量,+1表示加上了初始微服务
    A = [random.randint(2,6) for _ in range(K)]
    A = [x+1 for x in A]
    A = np.array(A)

    # 完全随机的层生成
    # 一共具有多少层
    # L = 80
    # # S_l = [random.uniform(10, 1000) for _ in range(L)]
    # S_l = [random.uniform(10, 1000) for _ in range(L)]
    # S_l = np.atleast_2d(np.array(S_l)).transpose()

    # E_kil = []
    # for k in range(K):
    #     # 对于每个应用程序，根据其微服务数量随机生成其所包含的层，并将其添加到E_kil中
    #     ms_number = A[k] # 应用程序k的微服务数量
    #     e_kil = np.zeros((ms_number, L))
    #     for ms in range(1,ms_number):
    #         # 对于每个微服务，随机生成其所包含的层
    #         layers = random.sample(range(L), random.randint(2,3)) #随机生成2-3层
    #         e_kil[ms, layers] = 1
    #     E_kil.append(e_kil)

    # 按照某些顺序进行层的生成
    L = sum(A)+10
    S_l = [random.uniform(10, 1000) for _ in range(L)]
    S_l = np.atleast_2d(np.array(S_l)).transpose()

    E_kil = []
    L_count = 0
    for k in range(K):
        # 对于每个应用程序，根据其微服务数量随机生成其所包含的层，并将其添加到E_kil中
        ms_number = A[k] # 应用程序k的微服务数量
        e_kil = np.zeros((ms_number, L))
        for ms in range(1,ms_number):
            # 对于每个微服务，随机生成其所包含的层
            layers = []
            layers.append(random.randint(0,10))
            layers.append(L_count+10)
            L_count += 1
            e_kil[ms, layers] = 1
        E_kil.append(e_kil)


    #目前的source是从0开始的
    Source = [random.randint(0, N-1) for _ in range(K)]

    u = []
    for k in range(K):
        u_k = [0 if ms == 0 else random.uniform(0.002, 0.5) for ms in range(A[k])]
        u.append(np.array(u_k).transpose())

    w = []
    for k in range(K):
        ms_number = A[k] # 应用程序k的微服务数量
        w_k = np.zeros((A[k],A[k]))
        for i in range(A[k]-1):
            w_k[i][i+1] = random.uniform(0.1, 2)
        w.append(w_k)

    D = get_path_matrix(create_topo(N))
    print("创建了一套随机微服务参数")
    para_dict = {}
    para_dict["K"] = K
    para_dict["N"] = N
    para_dict["C_S"] = C_S
    para_dict["C_C"] = C_C
    para_dict["A"] = A
    para_dict["L"] = L
    para_dict["E_kil"] = E_kil
    para_dict["S_l"] = S_l
    para_dict["u"] = u
    para_dict["w"] = w
    para_dict["D"] = D
    para_dict["b_cloud"] = b_cloud
    para_dict["Source"] = Source

    return K,N,C_S,C_C,A,L,E_kil,S_l,u,w,D,b_cloud,Source,para_dict

# 生成拓扑
def create_topo(N = 10):
    """
    create a topology with N nodes
    """
    candidate = [(1,2),(2,3),(2,10),(2,15),(3,4),(3,5),(3,14),(4,10),(6,7),(7,8),(8,9),(8,10),(4,7),(10,11),(11,12),(11,13)]
    actual = []
    for tup in candidate:
        if tup[0] <= N and tup[1] <= N:
            actual.append(tup)

    G = nx.Graph()
    G.add_nodes_from(range(1,N+1))
    G.add_edges_from(actual)
    return G

def draw_topo(G):
    nx.draw(G, with_labels=True)
    plt.show()

def get_path_matrix(G):
    """
    Return the shortest path matrix of the graph G
    """
    D_matrix = np.array([[0,1,2,3,3,5,4,3,4],
                         [1,0,1,2,2,4,3,2,3],
                         [2,1,0,1,1,3,2,3,4],
                         [3,2,1,0,2,2,1,2,3],
                         [3,2,1,2,0,4,3,4,5],
                         [5,4,3,2,4,0,1,2,3],
                         [4,3,2,1,3,1,0,1,2],
                         [3,2,3,2,4,2,1,0,1],
                         [4,3,4,3,5,3,2,1,0]])
    if len(G.nodes) < 10:
        return D_matrix[:len(G.nodes),:len(G.nodes)]
    path_matrix = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            if i == j:
                continue
            else:
                path = nx.shortest_path(G, i, j)
                path_matrix[i-1][j-1] = len(path)-1
    # print(path_matrix)
    return path_matrix

if __name__ == '__main__':
    G = create_topo()
    get_path_matrix(G)
    draw_topo(G)
    create_microservice(10,10)
    