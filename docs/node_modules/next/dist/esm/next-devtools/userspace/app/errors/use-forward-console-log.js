import { useEffect } from 'react';
import { isTerminalLoggingEnabled, logQueue } from '../forward-logs';
export const useForwardConsoleLog = (socketRef)=>{
    useEffect(()=>{
        if (!isTerminalLoggingEnabled) {
            return;
        }
        const socket = socketRef.current;
        if (!socket) {
            return;
        }
        const onOpen = ()=>{
            logQueue.onSocketReady(socket);
        };
        socket.addEventListener('open', onOpen);
        return ()=>{
            socket.removeEventListener('open', onOpen);
        };
    }, [
        socketRef
    ]);
};

//# sourceMappingURL=use-forward-console-log.js.map