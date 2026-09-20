import React from 'react';

/**
 * Single button primitive for the global navbar. All actions in the navbar
 * use this so heights, borders, padding, font, and rounding stay consistent
 * regardless of color (default / accent / primary).
 *
 *   <NavButton icon={Save} label="Save" onClick={...} />
 *   <NavButton icon={LogIn} label="Login" />
 *   <NavButton label="Sign Up" tone="accent" />
 */
const TONES = {
    default: 'df-button',
    primary: 'df-button df-button-primary',
    accent:  'df-button df-button-accent',
};

const NavButton = ({ icon: Icon, label, onClick, tone = 'default', title, children, ...buttonProps }) => (
    <button
        type="button"
        {...buttonProps}
        onClick={onClick}
        title={title ?? label}
        className={[
            TONES[tone] ?? TONES.default,
        ].join(' ')}
    >
        {Icon && <Icon size={13} strokeWidth={2.5} />}
        {label}
        {children}
    </button>
);

export default NavButton;
