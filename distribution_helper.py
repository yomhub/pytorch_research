from multiprocessing.managers import BaseManager
from multiprocessing import Queue
import os

class Master:
    def __init__(self,
        master_addr,port:int=8899,authkey:str='uf$hgf%'.encode("utf-8")):
        
        self.dispatched_task_queue = Queue()
        self.finished_task_queue = Queue()
        self.master_addr = master_addr
        self.port = port
        self.authkey = authkey
 
    def get_dispatched_task_queue(self):
        return self.dispatched_task_queue
 
    def get_finished_task_queue(self):
        return self.finished_task_queue

    def start(self,task_arg_list):
        BaseManager.register('get_dispatched_task_queue', callable=self.get_dispatched_task_queue)
        BaseManager.register('get_finished_task_queue', callable=self.get_finished_task_queue)

        manager = BaseManager(address=(self.master_addr, self.port), authkey=self.authkey.encode("utf-8"))
        manager.start()

        dispatched_tasks = manager.get_dispatched_task_queue()
        finished_tasks = manager.get_finished_task_queue()

        worker_pool = []
        processing_list = []
        while(task_arg_list):
            if(worker_pool):
                cur_worker = worker_pool.pop(0)
                dispatched_tasks.put(task_arg_list.pop(0))


class Slave:
    def __init__(self,
        master_addr,port:int=8899,authkey:str='uf$hgf%'.encode("utf-8")):
        self.master_addr = master_addr
        self.port = port
        self.authkey = authkey

    def start(self):
        manager = BaseManager(address=(self.master_addr, self.port), authkey=self.authkey.encode("utf-8"))
        manager.connect()
        