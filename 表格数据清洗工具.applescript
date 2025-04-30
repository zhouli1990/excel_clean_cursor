-- 表格数据清洗工具应用

-- 应用路径变量
property appPath : ""

-- 在应用启动时执行
on run
    -- 获取应用路径
    set appPath to (path to me as text)
    set appPath to POSIX path of appPath
    set appPath to do shell script "dirname " & quoted form of appPath
    
    -- 启动Flask服务
    startFlaskServer()
    
    -- 打开浏览器
    delay 1.5
    openBrowser()
end run

-- 在退出应用时执行
on quit
    -- 停止Flask服务
    stopFlaskServer()
    -- 继续退出过程
    continue quit
end quit

-- 启动Flask服务器
on startFlaskServer()
    -- 检查服务是否已运行
    set isRunning to checkIfRunning()
    if isRunning then
        display dialog "表格数据清洗工具服务已在运行中。" buttons {"确定"} default button 1
        openBrowser()
        return
    end if
    
    -- 使用bash调用Flask启动脚本
    set shellScript to "cd " & quoted form of appPath & " && ./run_flask_app.sh > /dev/null 2>&1 &"
    do shell script shellScript
    
    -- 显示成功消息
    display notification "表格数据清洗工具服务已启动" with title "服务启动"
end startFlaskServer

-- 检查服务是否已在运行
on checkIfRunning()
    try
        set result to do shell script "pgrep -f 'python app.py'"
        return true
    on error
        return false
    end try
end checkIfRunning

-- 停止Flask服务器
on stopFlaskServer()
    try
        do shell script "pkill -f 'python app.py'"
        display notification "表格数据清洗工具服务已停止" with title "服务停止"
    on error
        -- 忽略错误（可能是服务已经不在运行）
    end try
end stopFlaskServer

-- 打开浏览器
on openBrowser()
    -- 使用系统默认浏览器打开应用
    do shell script "open http://localhost:5100"
end openBrowser 