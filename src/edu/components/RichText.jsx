import { Fragment } from 'react';
import { interpolate } from '../lessons/engine/i18n.js';

/**
 * Lesson markup (docs/edu/SPEC.md §6.3.4): **bold**, `code`, blank-line paragraphs, "- " bullets and {name}
 * interpolation (numbers get locale grouping). Rendered as React elements; never as HTML.
 */
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`)/g;

function inline(text, keyPrefix) {
    return text.split(INLINE).filter((part) => part !== '').map((part, index) => {
        const key = `${keyPrefix}-${index}`;
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4) return <strong key={key}>{part.slice(2, -2)}</strong>;
        if (part.startsWith('`') && part.endsWith('`') && part.length > 2) return <code key={key} className="edu-code">{part.slice(1, -1)}</code>;
        return <Fragment key={key}>{part}</Fragment>;
    });
}

function block(paragraph, index) {
    const lines = paragraph.split('\n');
    const nodes = [];
    let bullets = [];
    const flush = () => {
        if (!bullets.length) return;
        nodes.push(<ul key={`ul-${index}-${nodes.length}`}>{bullets.map((line, item) => <li key={item}>{inline(line, `li-${index}-${item}`)}</li>)}</ul>);
        bullets = [];
    };
    let text = [];
    const flushText = () => {
        if (!text.length) return;
        nodes.push(<p key={`p-${index}-${nodes.length}`}>{inline(text.join(' '), `p-${index}-${nodes.length}`)}</p>);
        text = [];
    };
    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('- ')) { flushText(); bullets.push(trimmed.slice(2)); } else if (trimmed) { flush(); text.push(trimmed); }
    }
    flushText();
    flush();
    return nodes;
}

/** `inline` renders one line of inline markup without paragraph elements. */
export default function RichText({ text = '', params, inline: isInline = false, className }) {
    const source = params ? interpolate(String(text ?? ''), params) : String(text ?? '');
    if (isInline) return <span className={className}>{inline(source.replace(/\s*\n\s*/g, ' '), 'inline')}</span>;
    const paragraphs = source.split(/\n\s*\n/).filter((part) => part.trim());
    return <div className={['edu-rich', className].filter(Boolean).join(' ')}>{paragraphs.flatMap(block)}</div>;
}
