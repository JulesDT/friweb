copyright c 1998 free software foundation inc this file is part of the gnu c library contributed by zack weinberg zack rabi phys columbia edu 1998 the gnu c library is free software you can redistribute it and or modify it under the terms of the gnu lesser general public license as published by the free software foundation either version 2.1 of the license or at your option any later version the gnu c library is distributed in the hope that it will be useful but without any warranty without even the implied warranty of merchantability or fitness for a particular purpose see the gnu lesser general public license for more details you should have received a copy of the gnu lesser general public license along with the gnu c library if not write to the free software foundation inc 59 temple place suite 330 boston ma 02111 1307 usa include sys types h include termios h include unistd h include utmp h include pty h libutil_hidden_proto openpty libutil_hidden_proto login_tty int forkpty amaster name termp winp int amaster char name struct termios termp struct winsize winp int master slave pid if openpty &master &slave name termp winp 1 return 1 switch pid fork case 1 return 1 case 0 child close master if login_tty slave _exit 1 return 0 default parent amaster master close slave return pid