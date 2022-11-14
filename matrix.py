import utils
import numpy as np

def matrix(K=10,N=10, SCA = False):
    _,_,C_S,C_C,A,L,E_kil,S_l,u,w,D,b_cloud,Source,para_dict = utils.create_microservice(K,N)

    D_size = np.shape(D)[0]

    x_dim = sum(A)*N
    d_dim = N*L

    W_k = []
    for k in range(K):
        D_big = np.tile(D, (A[k], A[k]))
        w_k = np.zeros((A[k]*D_size, A[k]*D_size))
        for i in range(A[k]):
            for j in range(A[k]):
                w_k[i*D_size:(i+1)*D_size, j*D_size:(j+1)*D_size] = w[k][i][j]

        W_k.append(w_k*D_big)

    q = np.ones((1,N))
    Q_k = []
    for k in range(K):
        blk = []
        for i in range(A[k]):
            blk = blkdiag(blk, q)
        Q_k.append(blk)

    P = []
    for k in range(K):
        P_k = []
        for n in range(N):
            p_N_n = p_m_n(N,n)
            blk = []
            for i in range(A[k]):
                blk = blkdiag(blk, p_N_n)
            P_k.append(blk)
        P.append(P_k)

    V = []
    for k in range(K):
        V_k = []
        for l in range(L):
            p_L_n = p_m_n(L,l).transpose()
            blk = []
            for i in range(A[k]):
                blk = blkdiag(blk, p_L_n)
            V_k.append(blk)
        V.append(V_k)

    E = []
    for k in range(K):
        E_k = np.zeros((L*A[k],1))
        for i in range(A[k]):
            E_k[i*L:(i+1)*L] = np.atleast_2d(E_kil[k][i]).transpose()
        E.append(E_k)

    # 定义最后的大矩阵
    M = []
    for n in range(N):
        M = hstack(M, S_l.transpose()/b_cloud[n])

    W = []
    for k in range(K):
        W = blkdiag(W, W_k[k])

    b = np.ones((sum(A),1))

    Q = []
    for k in range(K):
        Q = blkdiag(Q, Q_k[k])

    Y_n = []
    for n in range(N):
        mid_M = []
        for k in range(K):
            mid_m = []
            for l in range(L):
                mid_m = hstack(mid_m,  P[k][n].dot(V[k][l]).dot(E[k]))
            mid_M = vstack(mid_M, mid_m)
        Y_n.append(mid_M)

    G_n = []
    for n in range(N):
        mid = []
        for k in range(K):
            mid = vstack(mid, np.atleast_2d(P[k][n].dot(u[k])).transpose())
        G_n.append(mid)

    G = []
    for n in range(N):
       G = hstack(G, G_n[n])
    G = G.transpose()

    Y = []
    for n in range(N):
        Y = hstack(Y, Y_n[n])
    Y = Y.transpose()


    S = []
    for n in range(N):
        S = blkdiag(S, S_l)
    S = S.transpose()

    #虚拟微服务的定义
    VG= []
    for k in range(K):
        VG_k = p_m_n(N,Source[k])
        VG = vstack(VG, VG_k)

    eye_n = np.eye(N)
    zero_n = np.zeros((N,N))
    H_k = []
    for k in range(K):
        mid = []
        for i in range(A[k]):
            if i == 0:
                mid = eye_n
            else:
                mid = hstack(mid, zero_n)
        H_k.append(mid)

    H = []
    for k in range(K):
        H = blkdiag(H, H_k[k])

    P_SCA = 0
    N_SCA = 0

    if SCA:
        W = W + W.T
        W_size = np.shape(W)[0]
        eig_w = np.linalg.eig(W)
        max_eig = max(abs(eig_w[0]))
        P_SCA = W + max_eig*np.eye(W_size)
        N_SCA = max_eig*np.eye(W_size)
        W = W/2

    para_dict["M"] = M
    para_dict["W"] = W
    para_dict["Q"] = Q
    para_dict["b"] = b
    para_dict["Y"] = Y
    para_dict["S"] = S
    para_dict["G"] = G
    para_dict["H"] = H
    para_dict["VG"] = VG
    para_dict["x_dim"] = x_dim
    para_dict["d_dim"] = d_dim
    para_dict["b"] = b
    para_dict["P_SCA"] = P_SCA
    para_dict["N_SCA"] = N_SCA

    return M,W,Q,b,Y,S,G,C_S,C_C,H,VG,x_dim,d_dim,P_SCA,N_SCA,para_dict

def blkdiag(a,b):
    """
    将输入的矩阵构造为分块对角矩阵
    """
    if not len(a):
        return b
    elif not len(b):
        return a
    a_shape = np.atleast_2d(a)
    b_shape = np.atleast_2d(b)
    return np.block([
        [a, np.zeros((a_shape.shape[0],b_shape.shape[1]))], 
        [np.zeros((b_shape.shape[0],a_shape.shape[1])), b]])

def p_m_n(m,n):
    # 该函数的作用是构造通用列向量p_m_n，向量长度为m，第n位为1，其余都为0
    p = np.zeros((m,1))
    p[n] = 1
    return p

def hstack(a,b):
    if not len(a):
        return b
    elif not len(b):
        return a
    return np.hstack((a,b))

def vstack(a,b):
    if not len(a):
        return b
    elif not len(b):
        return a
    return np.vstack((a,b))

if __name__ == '__main__':
    matrix(10,10)