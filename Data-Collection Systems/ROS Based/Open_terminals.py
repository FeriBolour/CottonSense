def open_terminals():
    import libtmux
    import os

    os.system("gnome-terminal -- /bin/sh -c 'tmux new-session -s ros -n New'")
    server = libtmux.Server()

    session = server.get_by_id('$0')

    window = session.new_window(attach=False, window_name="Cotton_img")
    session.kill_window("New")

    window.split_window(attach=False, vertical=False)

    pane0 = window.select_pane('0')

    pane1 = window.select_pane('1')

    return [pane0,pane1]