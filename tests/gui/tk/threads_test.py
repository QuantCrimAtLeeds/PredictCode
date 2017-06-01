import pytest
import unittest.mock as mock

import open_cp.gui.tk.threads as threads

def test_BackgroundTasks():
    root_mock = mock.MagicMock()
    b = threads.BackgroundTasks(root_mock)
    assert len(root_mock.after.call_args_list) == 1
    wait_time, func = root_mock.after.call_args_list[0][0]
    assert wait_time == 50
    
    x = []
    def task():
        x.append(5)
    
    b.submit(task)   
    
    assert x == []  
    func()
    assert x == [5]
    
@mock.patch("concurrent.futures.ThreadPoolExecutor")
@mock.patch("open_cp.gui.tk.threads.BackgroundTasks")
def test_Pool(bt_mock, executor_mock):
    bt_mock.return_value = mock.MagicMock()
    executor_mock.return_value = mock.MagicMock()
    root_mock = mock.MagicMock()
    pool = threads.Pool(root_mock)


    def task():
        return 55
    x = []
    def on_thread_task(value):
        x.append(value)
        
    pool.submit(task, on_thread_task)

    assert executor_mock.return_value.submit.call_count == 1
    internal_task = executor_mock.return_value.submit.call_args_list[0][0][0]
    internal_task()
    assert bt_mock.return_value.submit.call_count == 1
    gui_thread_task = bt_mock.return_value.submit.call_args_list[0][0][0]
    assert x == []
    gui_thread_task()
    assert x == [55]
    
def test_Pool_submit_Rules():
    root_mock = mock.MagicMock()
    pool = threads.Pool(root_mock)

    def task():
        pass

    with pytest.raises(ValueError):
        pool.submit(task, None)
    
    with pytest.raises(ValueError):
        class OurTask(threads.OffThreadTask):
            pass
        
        pool.submit(OurTask(), task)