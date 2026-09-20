/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Columns2 } from 'lucide-react';
import './open-to-side.css';

const OpenToSideContext = createContext(null);

function OpenToSideMenu({ menu, onClose, onOpen }) {
    const menuRef = useRef(null);
    const actionRef = useRef(null);

    useLayoutEffect(() => {
        const element = menuRef.current;
        const rect = element.getBoundingClientRect();
        const margin = 8;
        element.style.left = `${Math.max(margin, Math.min(menu.x, window.innerWidth - rect.width - margin))}px`;
        element.style.top = `${Math.max(margin, Math.min(menu.y, window.innerHeight - rect.height - margin))}px`;
        actionRef.current?.focus({ preventScroll: true });
    }, [menu]);

    useEffect(() => {
        const onPointerDown = (event) => {
            if (!menuRef.current?.contains(event.target)) onClose(false);
        };
        const dismiss = () => onClose(false);
        document.addEventListener('pointerdown', onPointerDown, true);
        window.addEventListener('resize', dismiss);
        window.addEventListener('blur', dismiss);
        return () => {
            document.removeEventListener('pointerdown', onPointerDown, true);
            window.removeEventListener('resize', dismiss);
            window.removeEventListener('blur', dismiss);
        };
    }, [onClose]);

    return createPortal(
        <div
            ref={menuRef}
            role="menu"
            aria-label="Level actions"
            className="df-open-side-menu nodrag nopan nowheel"
            style={{ left: menu.x, top: menu.y }}
            onContextMenu={(event) => event.preventDefault()}
            onBlur={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget)) onClose(false);
            }}
            onKeyDown={(event) => {
                event.stopPropagation();
                if (event.key === 'Escape') {
                    event.preventDefault();
                    onClose(true);
                } else if (event.key === 'Tab') {
                    onClose(true);
                } else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
                    event.preventDefault();
                    actionRef.current?.focus();
                } else if (event.altKey && event.key === 'Enter') {
                    event.preventDefault();
                    onOpen();
                }
            }}
        >
            <button ref={actionRef} type="button" role="menuitem" aria-keyshortcuts="Alt+Enter" onClick={onOpen}>
                <Columns2 size={14} strokeWidth={2.25} aria-hidden="true" />
                <span>Open to the side</span>
                <kbd>Alt+Enter</kbd>
            </button>
        </div>,
        document.body,
    );
}

/** The callback receives the absolute level path to open in the other pane. */
export function OpenToSideProvider({ value, children }) {
    const [menu, setMenu] = useState(null);
    const openMenu = useCallback((path, event) => {
        const target = event.currentTarget;
        const origin = target.matches?.('.react-flow__node, .df-level-tab')
            ? target
            : target.closest?.('.react-flow__node') ?? target.querySelector?.('.df-level-tab') ?? target;
        const rect = origin.getBoundingClientRect();
        const keyboard = event.type === 'keydown' || (!event.clientX && !event.clientY);
        setMenu({
            path: [...path],
            origin,
            x: keyboard ? rect.left : event.clientX,
            y: keyboard ? rect.bottom + 4 : event.clientY,
        });
    }, []);
    const closeMenu = useCallback((restoreFocus = false) => {
        if (restoreFocus && menu?.origin.isConnected) menu.origin.focus({ preventScroll: true });
        setMenu(null);
    }, [menu]);
    const context = useMemo(() => typeof value === 'function' ? { openToSide: value, openMenu } : null, [value, openMenu]);

    return (
        <OpenToSideContext.Provider value={context}>
            {children}
            {menu && context && (
                <OpenToSideMenu menu={menu} onClose={closeMenu} onOpen={() => {
                    closeMenu(true);
                    value(menu.path);
                }} />
            )}
        </OpenToSideContext.Provider>
    );
}

export function useOpenToSide() {
    return useContext(OpenToSideContext)?.openToSide ?? null;
}

export function useOpenToSideMenu(path, { disabled = false } = {}) {
    const context = useContext(OpenToSideContext);
    const available = Boolean(context && !disabled);
    const onContextMenu = useCallback((event) => {
        if (!available) return;
        event.preventDefault();
        event.stopPropagation();
        context.openMenu(path, event);
    }, [available, context, path]);
    const onKeyDown = useCallback((event) => {
        if (!available || event.defaultPrevented || event.target?.closest?.('input, textarea, select, [contenteditable="true"]')) return;
        if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.key === 'Enter') {
            event.preventDefault();
            event.stopPropagation();
            context.openToSide(path);
        } else if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) {
            onContextMenu(event);
        }
    }, [available, context, path, onContextMenu]);
    return { onContextMenu, onKeyDown, 'aria-keyshortcuts': available ? 'Alt+Enter' : undefined };
}

/** ReactFlow gives keyboard focus to the wrapper around a custom node. */
export function useOpenToSideNode(path, options) {
    const nodeRef = useRef(null);
    const menuProps = useOpenToSideMenu(path, options);
    const { onKeyDown } = menuProps;
    useEffect(() => {
        const wrapper = nodeRef.current?.closest('.react-flow__node');
        if (!wrapper) return;
        wrapper.addEventListener('keydown', onKeyDown, true);
        return () => wrapper.removeEventListener('keydown', onKeyDown, true);
    }, [onKeyDown]);
    return { ref: nodeRef, ...menuProps };
}
