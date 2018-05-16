#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex> 
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <iostream>
//#include <boost/thread/locks.hpp>
//#include <boost/thread/shared_mutex.hpp>
class ThreadPool {
public:
    int running_task_num;
    ThreadPool(size_t,std::condition_variable*, std::mutex*);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
private:
    //void change_num(int);
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;


    //typedef boost::shared_mutex Lock;                  
    //typedef boost::unique_lock< Lock >  WriteLock;
    //Lock myLocker;    
    
    
    // synchronization
    std::mutex* queue_mutex;
    std::condition_variable* this_condition;
    std::condition_variable* condition;
    std::mutex* out_mutex;
    bool stop;
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads,std::condition_variable* cond, std::mutex* mut)
    :   stop(false), running_task_num(0), condition(cond), out_mutex(mut)
{
    this_condition = new std::condition_variable;
    queue_mutex = new std::mutex;
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(*(this->queue_mutex));
                        this->this_condition->wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    //change_num(1);
		    {
			std::unique_lock<std::mutex> lock(*(this->out_mutex));
			++running_task_num;
		    }
		    condition->notify_one();
		    //std::cout<< "begin task" <<std::endl;
		    task();

		    {
			std::unique_lock<std::mutex> lock(*(this->out_mutex));
		   	--running_task_num;

		    //std::cout<< "end task" <<std::endl;
		    //std::cout<< "task num " << running_task_num<<std::endl;
		    }	
    		    condition->notify_one();
                    //change_num(-1);
		}
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(*(this->queue_mutex));

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task](){ (*task)(); });
    }
    this_condition->notify_one();
    return res;
}

/*
inline ThreadPool::change_num(int i)
{
	WriteLock wLock(myLocker);
	running_task_num += i;
}
*/



// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(*(this->queue_mutex));
        stop = true;
    }
    this_condition->notify_all();
    for(std::thread &worker: workers)
        worker.join();
}



#endif
