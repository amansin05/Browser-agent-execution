import type { ReactNode } from "react";

// A tiny, dependency-free markdown renderer for the agent's final shortlist (the synthesizer emits
// headings, **bold**, `code`, bullet/numbered lists, --- rules, and paragraphs). Renders to JSX —
// no dangerouslySetInnerHTML — so LLM output can't inject HTML/script.

function renderInline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) out.push(<strong key={key++}>{m[1]}</strong>);
    else if (m[2] !== undefined) out.push(<em key={key++}>{m[2]}</em>);
    else if (m[3] !== undefined) out.push(<code key={key++}>{m[3]}</code>);
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ content }: { content: string }) {
  const lines = (content || "").replace(/\r/g, "").split("\n");
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let list: ReactNode[] | null = null;
  let ordered = false;
  let key = 0;

  const flushPara = () => {
    if (para.length) { blocks.push(<p key={key++}>{renderInline(para.join(" "))}</p>); para = []; }
  };
  const flushList = () => {
    if (list) {
      blocks.push(ordered ? <ol key={key++}>{list}</ol> : <ul key={key++}>{list}</ul>);
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushPara(); flushList(); continue; }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushPara(); flushList();
      const level = Math.min(heading[1].length + 1, 6);  // shift down so page H1 stays largest
      const Tag = `h${level}` as keyof JSX.IntrinsicElements;
      blocks.push(<Tag key={key++}>{renderInline(heading[2])}</Tag>);
      continue;
    }
    if (/^([-*_])\1{2,}$/.test(line)) { flushPara(); flushList(); blocks.push(<hr key={key++} />); continue; }

    const bullet = /^[-*]\s+(.*)$/.exec(line);
    const numbered = /^\d+\.\s+(.*)$/.exec(line);
    if (bullet || numbered) {
      flushPara();
      const isOrdered = Boolean(numbered);
      if (list && ordered !== isOrdered) flushList();
      ordered = isOrdered;
      (list ||= []).push(<li key={key++}>{renderInline((bullet || numbered)![1])}</li>);
      continue;
    }
    flushPara();  // a non-list line ends any open list before starting a paragraph
    flushList();
    para.push(line);
  }
  flushPara();
  flushList();
  return <div className="md">{blocks}</div>;
}
