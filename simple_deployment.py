import numpy as np
import copy
from utils import create_microservice
import random 

class Deployment:
    def __init__(self,dict):
        """
        从matlab中读取数据并存储在python中
        """
        self.K = dict["K"]
        self.N = dict["N"]
        self.L = dict["L"]
        self.A = dict["A"]
        self.C_S = dict["C_S"]
        self.C_C = dict["C_C"]
        self.E_kil = dict["E_kil"]
        self.S_l = dict["S_l"].squeeze()
        self.u = dict["u"]
        self.w = dict["w"]
        self.Source = np.array(dict["Source"])+1
        self.b = dict["b_cloud"]
        self.D = dict["D"]
        
        self.max_hop = int(np.max(self.D))
        self.server_storage = self.C_S.copy()
        self.server_cpu = self.C_C.copy().astype(float) #数据类型

        # 服务器是否具有某一层
        self.server_layer = np.zeros((self.N,self.L))

        # deployment 对应的是真实的数字
        self.deployment = self.__build_deployment()

    def get_deployment(self):
        """
        获取部署策略
        """
        return self.deployment
            
    def __build_deployment(self):
        """
        构建deployment,格式与matlab中一致
        访问:deployment[k][i]
        """
        deployment = []
        for k in range(self.K):
            deployment_list = []
            for i in range(self.A[k]):
                if i == 0:
                    deployment_list.append(self.Source[k])
                else:
                    deployment_list.append(0)
            deployment.append(deployment_list)
        return deployment

    def set_deplotment(self, input):
        """
        人为设置部署策略,进行后续的处理
        """
        self.deployment = input

    def get_ms_resource(self,ms_k,ms_i):
        """
        获取微服务的请求资源，返回该微服务包含的层列表，请求的计算资源
        """
        l_list = []
        for l in range(self.L):
            if self.E_kil[ms_k][ms_i,l] == 1:
                l_list.append(l)
        return l_list,self.u[ms_k][ms_i]

    def get_ms_layer_list(self, ms):
        """
        获取微服务list的所有层和计算资源
        """
        l_list = []
        u_list = []
        for (ms_k, ms_i) in ms:
            if self.deployment[ms_k][ms_i] != 0:
                return False,False  # 已经被部署
            one_of_l_list, one_of_u = self.get_ms_resource(ms_k,ms_i)
            l_list.extend(one_of_l_list)
            u_list.append(one_of_u)
        l_list = list(set(l_list))
        return l_list, u_list

    def get_storage_from_layer(self,layer_list):
        """
        获取层列表所对应的所有存储资源
        """
        storage = 0
        for l in layer_list:
            storage += self.S_l[l]
        return storage

    def feasibility_of_deployment(self, ms, server_n):
        """
        :param ms: 微服务的编号列表,格式为[(k,i),(k,i),...]
        :param server_n: 服务器的编号
        判断微服务部署是否可行
        """
        l_list, u_list = self.get_ms_layer_list(ms)
        if l_list == False:
            return 1 # 已经被部署

        count_storage_new = 0
        for l in l_list:
            if self.server_layer[server_n][l] == 1:
                l_exist_in_server = 1
            else:
                count_storage_new += self.S_l[l]
        if count_storage_new > self.server_storage[server_n]:
            return 2 # 存储容量不足
        if sum(u_list) > self.server_cpu[server_n]:
            return 3 # 计算资源不足
        return [l_list,sum(u_list),count_storage_new] # 可以部署

    def deploy_ms(self, ms, server_n):
        """
        :param ms: 微服务的编号列表,格式为[(k,i),(k,i),...]
        :param server_n: 服务器的编号
        判断部署是否可行并将微服务部署到服务器上,返回0代表成功,否则为不同的错误代码
        更新服务器的存储容量，更新微服务的部署状态
        """
        #容错性检查
        if type(ms) == tuple:
            ms = [ms]

        feasibility = self.feasibility_of_deployment(ms, server_n)
        if type(feasibility) == int:
            return feasibility
        l_list, u, count_storage_new = feasibility[0], feasibility[1], feasibility[2]

        # 进行微服务部署
        # +1对应实际的服务器数值
        for (ms_k, ms_i) in ms:
            self.deployment[ms_k][ms_i] = server_n+1

        for l in l_list:
            self.server_layer[server_n][l] = 1
        self.server_storage[server_n] -= count_storage_new
        self.server_cpu[server_n] -= u

        return 0

    def deploy_ms_in_server_list(self, ms, server_list):
        """
        根据给定的服务器列表部署微服务
        """
        for server_n in server_list:
            deploy_result = self.deploy_ms(ms, server_n)
            if deploy_result == 0:
                return 0
        return deploy_result

    def find_ms_from_l(self,l):
        """
        根据层查找微服务，哪些微服务具有这一层
        """
        ms_list = []
        for k in range(self.K):
            for i in range(self.A[k]):
                if self.E_kil[k][i,l] == 1:
                    ms_list.append((k,i))
        return ms_list
        
    def find_close_server(self,server_n,hop):
        """
        找到服务器n的hop跳可达服务器
        """
        server_list = []
        for n in range(self.N):
            if self.D[server_n,n] == hop:
                server_list.append(n)
        return server_list

    def find_deployment_from_ms(self, ms):
        """
        找到ms列表中哪些微服务已经被部署
        """
        deployment_ms = []
        for (ms_k, ms_i) in ms:
            if self.check_ms_deployment(ms_k,ms_i):
                deployment_ms.append((ms_k,ms_i))
        return deployment_ms

    def get_resource_from_server(self,server_n):
        """
        获取服务器n的剩余资源
        """
        return self.server_storage[server_n],self.server_cpu[server_n]

    def get_deployment_server(self,ms_k,ms_i):
        """
        获取某个微服务的部署服务器
        """
        return self.deployment[ms_k][ms_i]-1

    def get_deployment_servers(self, ms):
        """
        获取微服务列表的部署服务器列表
        """
        deployment_servers = []
        for (ms_k, ms_i) in ms:
            deployment_servers.append(self.get_deployment_server(ms_k,ms_i))
        return deployment_servers

    def check_app_deployment(self,ms_k):
        """
        检查某个应用的部署情况,True代表有应用部署,False代表没有应用部署
        """
        for i in range(self.A[ms_k]):
            if i == 0:
                continue
            elif self.deployment[ms_k][i] != 0:
                return True
        return False

    def check_ms_deployment(self,ms_k,ms_i):
        """
        检查某个微服务的部署情况,True代表有微服务部署,False代表没有微服务部署
        """
        if ms_k > self.K or ms_i > self.A[ms_k]:
            raise ValueError('ms_k or ms_i is out of range')
        if self.deployment[ms_k][ms_i] != 0:
            return True
        return False

    def check_all_deployment(self):
        """
        检查所有微服务的部署情况,True代表都部署了,False代表没有部署
        """
        for k in range(self.K):
            for i in range(self.A[k]):
                if self.deployment[k][i] == 0:
                    return False
        return True

    def calculate_storage_occupation(self):
        """
        检查服务器的存储资源占用情况,部署完成后才能调用
        """
        if self.check_all_deployment() == False:
            raise ValueError('not all deployment')
        # remain_storage = 0
        # for n in range(self.N):
        #     remain_storage += self.server_storage[n]
        # return np.sum(self.C_S) - remain_storage
        storage = 0
        for n in range(self.N):
            storage += np.matmul(self.server_layer[n],self.S_l.transpose())/1000
        return storage

    def calculate_download_time(self):
        """
        检查服务器的下载时间,部署完成后才能调用
        """
        if self.check_all_deployment() == False:
            raise ValueError('not all deployment')
        download_time = 0
        for n in range(self.N):
            download_time += np.matmul(self.server_layer[n],self.S_l.transpose())/self.b[n]
        return download_time
        
    def calculate_communication(self):
        """
        检查服务器的通讯开销,部署完成后才能调用
        """
        if self.check_all_deployment() == False:
            raise ValueError('not all deployment')
        communication = 0
        for k in range(self.K):
            for i in range(self.A[k]):
                for j in range(self.A[k]):
                    server_1 = self.deployment[k][i]-1
                    server_2 = self.deployment[k][j]-1
                    hop = self.D[server_1,server_2]
                    communication += self.w[k][i,j]*hop
        return communication

    def deployment_encoder(self):
        """
        将部署结果编码为one-hot,部署完成后才能调用
        [[1, 6, 8], [2, 3, 5], [3, 9, 1, 5, 2], [4, 4, 6, 5, 1, 2], [5, 2, 4, 9], [6, 6, 2], [7, 8, 7, 9], [8, 3, 2, 8], [9, 7, 7, 6, 4]]
        """
        if self.check_all_deployment() == False:
            raise ValueError('not all deployment')
        x_encoder = []
        for k in self.deployment:
            for i in k:
                temp = [0] * self.N
                temp[i-1] = 1
                x_encoder.extend(temp)
        #why? 不加int相乘会变成59？
        d_encoder = self.server_layer.reshape(int(self.L)*int(self.N))
        return x_encoder, d_encoder

    def generate_init_layer(self):
        """
        对每台服务器都生成一些初始就有的层，让后面的差距起来
        """
        for n in range(self.N):
            for l in random.sample([n for n in range(self.L)], 1):
                self.server_layer[n][l] = 1

    #下面两个用于检查原本的通讯和空间占用
    def calculate_storage_total(self):
        sum = 0
        for k in range(self.K):
            for i in range(self.A[k]):
                for l in range(self.L):
                    sum += self.E_kil[k][i,l]*self.S_l[l]
        return sum

    def calculate_communication_total(self):
        sum = 0
        for k in range(self.K):
            for i in range(self.A[k]):
                for j in range(self.A[k]):
                    sum += self.w[k][i,j]
        return sum

class Simple_deployment(Deployment):
    """
    具体的部署算法相关的函数
    """
    def __init__(self,para_dict, theta = 0.5):
        super().__init__(para_dict)
        # self.generate_init_layer()
        self.theta = theta

        #归一化的通讯数据量和层大小
        self.w_weighted,self.l_weighted = self.weight_normalization(theta)

        # 构建出排序完成后的通讯数据和层大小
        self.data_linklist = self.sort()

    def weight_normalization(self,theta):
        """
        归一化权重
        """
        # w归一化
        max_w = 0
        min_w = 10000
        w_weighted = copy.deepcopy(self.w)

        for k in range(self.K):
            # 这个地方好像有个警告
            try:
                dense_array = self.w[k].todense()
            except:
                dense_array = self.w[k]
            mask_array = np.ma.masked_array(dense_array, mask=dense_array == 0)
            if np.min(mask_array) < min_w:
                min_w = np.min(mask_array)
            if np.max(mask_array) > max_w:
                max_w = np.max(mask_array)
        # 避免出现0的情况
        min_w = min_w * 0.9
        for k in range(self.K):
            for i in range(self.A[k]):
                for j in range(self.A[k]):
                    if self.w[k][i,j] != 0:
                        w_weighted[k][i,j] = theta*(self.w[k][i,j] - min_w)/(max_w - min_w)

        # S归一化
        max_S = 0
        min_S = 10000
        l_weighted = self.S_l.copy()
        for l in range(self.L):
            if self.S_l[l] > max_S:
                max_S = self.S_l[l]
            if self.S_l[l] < min_S:
                min_S = self.S_l[l]
        # 避免出现0的情况
        min_S = min_S * 0.9
        for l in range(self.L):
            l_weighted[l] = (1-theta)*(self.S_l[l] - min_S)/(max_S - min_S)

        return w_weighted,l_weighted

    def sort(self):
        """
        排序得出一些列表
        """
        # self.w_weighted[k][i,j] is np.ma.masked
        # w_weighted[k][i,j]
        # l_weighted[l]
        data_linklist = LinkedList()
        for k in range(self.K):
            for i in range(self.A[k]):
                for j in range(self.A[k]):
                    if self.w_weighted[k][i,j] != 0:
                        ms = [(k,i),(k,j)]
                        data_linklist.add(data_form=0,ms = ms,data = self.w_weighted[k][i,j])

        for l in range(self.L):
            ms = self.find_ms_from_l(l)
            data_linklist.add(data_form=1,ms = ms,data = self.l_weighted[l])
        # 对链表进行排序
        sort_result = data_linklist.sortList(data_linklist.head)
        data_linklist.head = sort_result
        data_linklist.cur = data_linklist.head
        # 测试用
        # print(data_linklist.count())

        return data_linklist
        
    def find_index_storage_server(self,i):
        """
        找到存储第i多的服务器
        """
        return np.argsort(self.server_storage)[-1*i]

    def get_max_index_from_server_list(self,server_list,resource = "storage"):
        """
        获取服务器列表中的最大资源服务器序号
        """
        if server_list == []:
            return -1
        max_storage = 0
        max_storage_server = -1
        max_cpu = 0
        max_cpu_server = -1
        for server_n in server_list:
            storage,cpu = self.get_resource_from_server(server_n)
            if storage > max_storage:
                max_storage = storage
                max_storage_server = server_n
            if cpu > max_cpu:
                max_cpu = cpu
                max_cpu_server = server_n
        if resource == "storage":
            return max_storage_server
        else:
            return max_cpu_server

    def get_max_index_list_from_server_list(self,server_list,resource = "storage"):
        """
        获取服务器列表中的按照资源大小排序后的列表
        """
        sort_list = []
        resource_list = []
        for server_n in server_list:
            if resource == "storage":
                resource_data,_ = self.get_resource_from_server(server_n)
            else:
                _,resource_data = self.get_resource_from_server(server_n)
            if sort_list == []:
                sort_list.append(server_n)
                resource_list.append(resource_data)
            else:
                for i in range(len(sort_list)):
                    if resource_data > resource_list[i]:
                        sort_list.insert(i,server_n)
                        resource_list.insert(i,resource_data)
                        break
                    elif i == len(sort_list)-1:
                        sort_list.append(server_n)
                        resource_list.append(resource_data)
                        break
        return sort_list

    def get_max_index_list_from_server(self,server_n,hop,resource = "storage"):
        """
        给定服务器n,获取服务器n的hop跳邻居中按照资源排序的服务器列表
        """
        hop_server_list = self.find_close_server(server_n,hop)
        return self.get_max_index_list_from_server_list(hop_server_list, resource=resource)

    def get_max_b_list_from_server_list(self,server_list):
        """
        获取服务器列表中的按照b排序后的列表
        """
        sort_list = []
        b_list = []
        for server_n in server_list:
            b_data = self.b[server_n]
            if sort_list == []:
                sort_list.append(server_n)
                b_list.append(b_data)
            else:
                for i in range(len(sort_list)):
                    if b_data > b_list[i]:
                        sort_list.insert(i,server_n)
                        b_list.insert(i,b_data)
                        break
                    elif i == len(sort_list)-1:
                        sort_list.append(server_n)
                        b_list.append(b_data)
                        break
        return sort_list
            
    def check_ms_list_deployment(self, ms):
        """
        检查ms列表中所有微服务的部署情况,所有服务都部署、部分部署和都没部署返回不同的标志
        """
        ms_len = len(ms)
        count = 0
        for (ms_k, ms_i) in ms:
            if self.check_ms_deployment(ms_k, ms_i):
                count += 1
        if count == ms_len:
            return 0 # 所有服务都部署
        elif count == 0:
            return 1 # 没有服务部署
        else:
            return 2 # 部分服务部署

    def search_nearest_deployment_ms(self, ms_k, ms_i):
        """
        搜索最近的部署微服务
        """
        ms_num = self.A[ms_k]
        for i in range(max(ms_num-ms_i,ms_i)):
            # 理论上可以优化为跟随上下游里面数据量更大的部署，后期再考虑
            up = ms_i + i + 1
            down = ms_i - i - 1
            if down >= 0:
                if self.check_ms_deployment(ms_k, down):
                    return (ms_k, down)
            if up < self.A[ms_k]:
                if self.check_ms_deployment(ms_k, up):
                    return (ms_k, up)
        return None

    def find_to_deploy(self, ms, server_n):
        """
        找到可部署的服务器并进行部署,根据多跳次数最近的原则和存储空间进行排序
        :param ms: 待部署的微服务/微服务列表
        :param server_n: 待部署的服务器
        :return: 成功则返回部署的服务器编号，失败则返回-1
        """
        for hop in range(self.max_hop):
            hop_server_list = self.get_max_index_list_from_server(server_n,hop,resource="storage")
            for hop_server in hop_server_list:
                # 遍历服务器部署微服务
                deploy_result = self.deploy_ms(ms, hop_server)
                if deploy_result == 0:
                    return hop_server
        return -1

    def split_and_deploy_chain(self, ms, server_n):
        """
        将微服务对拆分并进行部署
        """
        first_ms = ms[0]
        first_deploy_server = self.find_to_deploy(first_ms, server_n)

        if first_deploy_server == -1:
            # 没有找到可以部署的服务器
            raise Exception("no server can deploy")

        last_ms = ms[1]
        last_deploy_server = self.find_to_deploy(last_ms, first_deploy_server)

        if last_deploy_server == -1:
            # 没有找到可以部署的服务器
            raise Exception("no server can deploy")
        return True

    def split_and_deploy_layer(self, ms, server_list):
        """
        将微服务层拆分并进行部署
        """
        deploy_result = -1
        to_deploy_ms = []
        ms_list = ms.copy()

        while deploy_result != 0:
            single_ms, ms_list = self.best_split(ms_list)
            to_deploy_ms.append(single_ms)
            if len(ms_list) == 1:
                # 所有微服务都拆分成了单个的微服务
                to_deploy_ms.append(ms_list)
                break
            deploy_result = self.deploy_ms_in_server_list(ms_list, server_list)
        
        for ms in to_deploy_ms:
            deploy_result = self.deploy_ms_in_server_list(ms, server_list)
            if deploy_result != 0:
                raise Exception("deploy failed")
        return 0
    


    def best_split(self, ms):
        """
        对一组给定的微服务,拆分出一个微服务,使得拆分后占用的总空间最小
        """
        min_storage = 100000
        best_ms = None
        best_single_ms = None
        for single_ms in ms:
            # 对每一个微服务进行拆分,比较得出使得总占用空间最小的微服务拆分组合
            split_list = ms.copy()
            split_list.remove(single_ms)
            l_list1, _ = self.get_ms_layer_list(split_list)
            l_list2, _ = self.get_ms_layer_list([single_ms])
            storage = self.get_storage_from_layer(l_list1) + self.get_storage_from_layer(l_list2)
            if storage < min_storage:
                min_storage = storage
                best_ms = split_list
                best_single_ms = single_ms
            
        return [best_single_ms], best_ms

class ListNode:
    #链节点类，用于存储归一化数据
    def __init__(self,data_form = None,ms = None,data = None,next = None):
        self.data_form = data_form #数据形式为0代表是通讯数据，1代表是层大小
        self.ms = ms
        self.data = data
        self.next = next

class LinkedList:
    # 链表类，用于将微服务归一化后的权重进行排序
    def __init__(self):
        self.head = None
        self.cur = None

    def is_empty(self):
        return self.head == None

    def add(self,data_form,ms,data):
        if self.head == None:
            self.head = ListNode(data_form,ms,data)
            self.cur = self.head
        else:
            node = ListNode(data_form,ms,data)
            self.cur.next = node
            self.cur = node

    def sort(self):
        """
        冒泡排序
        """
        #给出一个新的头结点
        new_head = ListNode()

    def merge(self,linklist1,linklist2):
        """
        合并两个链表
        """
        merge_list = linklist = ListNode()

        while linklist1 and linklist2:
            if linklist1.data > linklist2.data:
                linklist.next, linklist, linklist1 = linklist1, linklist1, linklist1.next
            else:
                linklist.next, linklist, linklist2 = linklist2, linklist2, linklist2.next

        linklist.next = linklist1 or linklist2
        return merge_list.next
                
    
    def sortList(self, head):
        """
        对链表进行排序
        """
        if head == None or head.next == None:
            return head
        else:
            prev, slow, fast = None, head, head

        while fast and fast.next:
            prev, slow, fast = slow, slow.next, fast.next.next

        prev.next = None
        l1 = self.sortList(head)
        l2 = self.sortList(slow)
        return self.merge(l1, l2)

    def count(self):
        """
        计算链表的长度
        """
        cur = self.head
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

def simple_deployment_main(theta,para_dict)->Simple_deployment:
    """
    简单的部署策略
    """
    deployment = Simple_deployment(theta = theta, para_dict = para_dict)
    cur = deployment.data_linklist.head
    while cur != None:
        ms_deployment_flag = deployment.check_ms_list_deployment(cur.ms)
        if ms_deployment_flag == 0:
            # 所有微服务都部署
            pass
        elif ms_deployment_flag == 1:
            # 没有微服务部署
            if cur.data_form == 0:
                # 通讯数据
                # 拿到最近的部署服务器
                cloest_ms = deployment.search_nearest_deployment_ms(cur.ms[0][0], cur.ms[0][1])
                if cloest_ms != None:
                    server_to_deploy = deployment.get_deployment_server(cloest_ms[0], cloest_ms[1])
                    # 从0跳服务器开始部署
                    deploy_result = deployment.find_to_deploy(cur.ms, server_to_deploy)
                    # 成对部署失败的话拆分微服务进行部署
                    if deploy_result == -1:
                        deployment.split_and_deploy_chain(cur.ms, server_to_deploy)
            else:
                # 层大小数据
                server_list = [n for n in range(deployment.N)]
                sort_list = deployment.get_max_b_list_from_server_list( server_list)
                # 尝试能不能直接部署
                deploy_result = deployment.deploy_ms_in_server_list(cur.ms, sort_list)
                if deploy_result != 0:
                    # 不能直接部署的话就拆分微服务进行部署
                    deployment.split_and_deploy_layer(cur.ms, sort_list)
        else:
            # 部分微服务部署
            if cur.data_form == 0:
                # 通讯数据
                ms_to_deploy = cur.ms.copy()
                deployment_ms = deployment.find_deployment_from_ms(cur.ms)[0]
                ms_to_deploy.remove(deployment_ms)
                
                server_to_deploy = deployment.get_deployment_server(deployment_ms[0], deployment_ms[1])

                deploy_result = deployment.find_to_deploy(ms_to_deploy, server_to_deploy)
                if deploy_result == -1:
                    raise Exception("部署失败")
            else:
                # 层大小数据
                ms_to_deploy = cur.ms.copy()
                deployment_ms = deployment.find_deployment_from_ms(cur.ms)
                for ms in deployment_ms:
                    ms_to_deploy.remove(ms)
                servers_to_deploy = deployment.get_deployment_servers(deployment_ms)
                sort_list = deployment.get_max_b_list_from_server_list( servers_to_deploy)

                deploy_result = deployment.deploy_ms_in_server_list(ms_to_deploy, sort_list)
                if deploy_result != 0:
                    # 不能直接部署的话就扩大范围进行部署
                    server_list = [n for n in range(deployment.N)]
                    sort_list = deployment.get_max_b_list_from_server_list(server_list)
                    deploy_result = deployment.deploy_ms_in_server_list(ms_to_deploy, sort_list)
                if deploy_result != 0:
                    #还不能部署就拆分微服务进行部署
                    deployment.split_and_deploy_layer(ms_to_deploy, sort_list)
        cur = cur.next
        if deployment.check_all_deployment():
            break
    # return deployment.get_deployment(), deployment.calculate_storage_occupation(), deployment.calculate_communication()
    return deployment

def random_deployment(para_dict)->Simple_deployment:
    """
    随机部署策略
    """
    import random
    deployment = Deployment(para_dict)
    for k in range(deployment.K):
        for i in range(1,deployment.A[k]):
            random_server = random.sample([n for n in range(deployment.N)],deployment.N)
            for server_n in random_server:
                if deployment.deploy_ms(ms=(k,i), server_n=server_n) == 0:
                    break
    return deployment

class Layer_match_deployment(Deployment):
    """
    层匹配部署策略
    """
    def __init__(self, para_dict):
        super().__init__(para_dict)
        self.generate_init_layer()

    def get_server_list_through_layers(self,layers):
        """
        根据给定的微服务层，查询包含这些层大小最多的边缘服务器列表
        """
        layers_in_server = np.zeros(self.N)
        for n in range(self.N):
            for layer in layers:
                if self.server_layer[n][layer]:
                    # layers_in_server[n] += self.S_l[layer]
                    layers_in_server[n] += 1
        # 将已有的层按照总大小从大到小排序得出服务器列表
        sort_list = np.argsort(layers_in_server)[::-1]
        return sort_list

    def generate_init_layer(self):
        """
        对每台服务器都生成一些初始就有的层，让后面的差距起来
        """
        for n in range(self.N):
            for l in random.sample([n for n in range(self.L)], 1):
                self.server_layer[n][l] = 1

def layer_match_deployment(para_dict):
    """
    层匹配部署策略,对于每个容器，选择一个本地存储的图像层数量最多的边缘节点，并根据分配顺序对层进行排序。
    """
    deployment = Layer_match_deployment(para_dict)
    for k in range(deployment.K):
        for i in range(1,deployment.A[k]):
            layer, _ = deployment.get_ms_resource(k,i)
            sort_list = deployment.get_server_list_through_layers(layer)
            deploy_result = deployment.deploy_ms_in_server_list((k,i),sort_list)
            if deploy_result != 0:
                raise ValueError("部署失败")
    return deployment

class kubernetes_deployment(Deployment):
    """
    k8s的部署策略
    """
    def __init__(self, para_dict):
        super().__init__(para_dict)
        # self.generate_init_layer()

    def get_server_list_through_layers(self,layers):
        """
        根据给定的微服务层，查询包含这些层大小最多的边缘服务器列表
        """
        layers_in_server = np.zeros(self.N)
        for n in range(self.N):
            for layer in layers:
                if self.server_layer[n][layer]:
                    layers_in_server[n] += self.S_l[layer]
                    b = self.S_l[layer]
                    a = 1
        # 将已有的层按照总大小从大到小排序得出服务器列表
        sort_list = np.argsort(layers_in_server)[::-1]
        return sort_list

    def generate_init_layer(self):
        """
        对每台服务器都生成一些初始就有的层，让后面的差距起来
        """
        for n in range(self.N):
            for l in random.sample([n for n in range(self.L)], 1):
                self.server_layer[n][l] = 1

def k8s_deployment(para_dict):
    """
    k8s默认部署策略,对于每个容器，选择一个本地存储的图像层大小最多的边缘节点
    """
    deployment = kubernetes_deployment(para_dict)
    for k in range(deployment.K):
        for i in range(1,deployment.A[k]):
            layer, _ = deployment.get_ms_resource(k,i)
            sort_list = deployment.get_server_list_through_layers(layer)
            deploy_result = deployment.deploy_ms_in_server_list((k,i),sort_list)
            if deploy_result != 0:
                raise ValueError("部署失败")
    return deployment

if __name__ == "__main__":
    result = create_microservice(25,15)
    para_dict = result[-1]

    deployment = simple_deployment_main(theta = 0.5,para_dict = para_dict)
    # deployment = random_deployment(from_mat_or_matlab=1, K = 4, N = 4)
    print("simple deployment:",deployment.get_deployment())
    print("storage:",deployment.calculate_storage_occupation())
    print("download time:",deployment.calculate_download_time())
    print("communication:",deployment.calculate_communication())
    print("------------------------")

    deployment = layer_match_deployment(para_dict)
    print("layer match deployment:",deployment.get_deployment())
    print("storage:",deployment.calculate_storage_occupation())
    print("download time:",deployment.calculate_download_time())
    print("communication:",deployment.calculate_communication())
    print("------------------------")

    deployment = k8s_deployment(para_dict)
    print("k8s deployment:",deployment.get_deployment())
    print("storage:",deployment.calculate_storage_occupation())
    print("download time:",deployment.calculate_download_time())
    print("communication:",deployment.calculate_communication())
    print("------------------------")


    # deployment = random_deployment()
    # print(deployment.get_deployment())
    # print(deployment.calculate_storage_occupation())
    # print(deployment.calculate_communication())

#TODO: 在这个基础上可以进行进一步的改进，微服务/应用可以是手动一个一个增加上来的，然后增加一个部署一个，尽可能最小化整体的开销；然后还要支持应用程序的增删