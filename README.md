## README
* 数据结构请查看scheduling_env.py

## Algorithm
* 我能想到的最简单实现方法
* while true
    * 新来了Job
        * 冻结当前时间T
        * 取出所有机器的waiting队列 + 当前Job
        * 清空所有机器的waiting队列
        * 计算响应比，n个
            * 取出job1 + T -> State -> Action -> reward, next-state, done
            * 取出job2 + T -> State -> Action -> reward, next-state, done
            * ...
            * 取出jobn + T -> State -> Action -> reward, next-state, done
    * 来下一个Job
    * ...
    * 直到done，即100个任务全部跑完，break    
        
* done:来的是最后一个任务，排的是waiting的最后一个
    * T > env.jobs[-1].T_arrival