-- ���������ϴ����Ӧ��

-- Ӧ��·������
property appPath : "/Users/zhouli/Downloads/excel_clean_cursor"

-- ��Ӧ������ʱִ��
on run
    -- ��ȡӦ��·��
    set appPath to (path to me as text)
    set appPath to POSIX path of appPath
    set appPath to do shell script "dirname " & quoted form of appPath
    
    -- ����Flask����
    startFlaskServer()
    
    -- �������
    delay 1.5
    openBrowser()
end run

-- ���˳�Ӧ��ʱִ��
on quit
    -- ֹͣFlask����
    stopFlaskServer()
    -- �����˳�����
    continue quit
end quit

-- ����Flask������
on startFlaskServer()
    -- �������Ƿ�������
    set isRunning to checkIfRunning()
    if isRunning then
        display dialog "���������ϴ���߷������������С�" buttons {"ȷ��"} default button 1
        openBrowser()
        return
    end if
    
    -- ʹ��bash����Flask�����ű�
    set shellScript to "cd " & quoted form of appPath & " && ./run_flask_app.sh > /dev/null 2>&1 &"
    do shell script shellScript
    
    -- ��ʾ�ɹ���Ϣ
    display notification "���������ϴ���߷���������" with title "��������"
end startFlaskServer

-- �������Ƿ���������
on checkIfRunning()
    try
        set result to do shell script "pgrep -f 'python app.py'"
        return true
    on error
        return false
    end try
end checkIfRunning

-- ֹͣFlask������
on stopFlaskServer()
    try
        do shell script "pkill -f 'python app.py'"
        display notification "���������ϴ���߷�����ֹͣ" with title "����ֹͣ"
    on error
        -- ���Դ��󣨿����Ƿ����Ѿ��������У�
    end try
end stopFlaskServer

-- �������
on openBrowser()
    -- ʹ��ϵͳĬ���������Ӧ��
    do shell script "open http://localhost:5100"
end openBrowser 