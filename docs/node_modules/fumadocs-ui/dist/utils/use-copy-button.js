'use client';
import { useEffect, useRef, useState } from 'react';
import { useEffectEvent } from 'fumadocs-core/utils/use-effect-event';
export function useCopyButton(onCopy) {
    const [checked, setChecked] = useState(false);
    const timeoutRef = useRef(null);
    const onClick = useEffectEvent(() => {
        if (timeoutRef.current)
            window.clearTimeout(timeoutRef.current);
        const res = Promise.resolve(onCopy());
        void res.then(() => {
            setChecked(true);
            timeoutRef.current = window.setTimeout(() => {
                setChecked(false);
            }, 1500);
        });
    });
    // Avoid updates after being unmounted
    useEffect(() => {
        return () => {
            if (timeoutRef.current)
                window.clearTimeout(timeoutRef.current);
        };
    }, []);
    return [checked, onClick];
}
