import os, sys
#当前目录
dir_path = os.path.dirname(os.path.abspath(__file__))

# 并行工作进程数
workers = 1
# 指定每个工作者的线程数
threads = 3
# 监听内网端口80
bind = '0.0.0.0:5000'
# 工作模式协程
worker_class = 'gevent'
# 设置最大并发量
worker_connections = 10
# 设置进程文件目录
#pidfile = 'gunicorn.pid'
# 设置访问日志和错误信息日志路径
accesslog = '{}/logs/gunicorn_acess.log'.format(dir_path)
errorlog = '{]/logs/gunicorn_error.log'.format(dir_path)
# 设置日志记录水平
# loglevel = 'DEBUG'
# 代码发生变化是否自动重启
#reload=True

#preload_app = True
