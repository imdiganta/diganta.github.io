


        // function escapeHtml(unsafe) {
        //     return unsafe
        //         .replace(/&/g, "&amp;")
        //         .replace(/</g, "&lt;")
        //         .replace(/>/g, "&gt;")
        //         .replace(/"/g, "&quot;")
        //         .replace(/'/g, "&#039;");
        // }

        // function formatCode(text) {
        //     // Regular expressions for different patterns
        //     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
        //     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
        //     const headingPattern = /^(#+)\s*(.*)$/gm; // Headings (e.g., #, ##, ###)
        //     const bulletPointPattern = /^\s*\*\s+(.*)$/gm; // Bullet points (e.g., * item)
        //     const subBulletPointPattern = /^\s*\*\*\s+(.*)$/gm; // Subpoints (e.g., ** sub-item)
        //     const numberedPointPattern = /^\s*\d+\.\s+(.*)$/gm; // Numbered points (e.g., 1. item)
        //     const italicPattern = /_(.*?)_/g; // Italics (e.g., _text_)
        //     const boldPattern = /\*\*(.*?)\*\*/g; // Bold (e.g., **text**)
        //     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough (e.g., ~~text~~)
        //     const blockquotePattern = /^>\s*(.*)$/gm; // Blockquotes (e.g., > quote)
        //     const horizontalRulePattern = /^(---|___|\*\*\*)$/gm; // Horizontal rules (e.g., ---)
        //     const imagePattern = /!\[([^[]+)\]\(([^)]+)\)/g; // Images (e.g., ![alt text](url))
        //     const linkPattern = /\[([^[]+)\]\(([^)]+)\)/g; // Links (e.g., [text](url))
        //     const tablePattern = /\|([^\n]+)\|\n\|[-:|]+\|\n((\|[^\n]+\|)+)/g; // Tables
        //     const taskListPattern = /^\s*\[([ x])\]\s+(.*)$/gm; // Task lists (e.g., [x] item)
        //     const definitionListPattern = /^(.*?)\n:(.*)$/gm; // Definition lists
        //     const footnotePattern = /\[\^([^\]]+)\]:\s*(.*)/g; // Footnotes
        
        //     // Replace block code with <pre><code> tags
        //     text = text.replace(blockCodePattern, (match, p1) => {
        //         return `<pre><code>${escapeHtml((p1.replace(/<br>/g, '')).trim())}</code></pre>`;
        //     });
        
        //     // Replace inline code
        //     text = text.replace(inlineCodePattern, (match, p1) => {
        //         return `<code>${escapeHtml((p1.replace(/<br>/g, '')).trim())}</code>`;
        //     });
        
        //     // Replace headings
        //     text = text.replace(headingPattern, (match, hashes, content) => {
        //         const level = hashes.length; // Number of '#' determines heading level
        //         return `<h${level}>${escapeHtml(content.trim())}</h${level}>`;
        //     });
        
        //     // Replace bold text
        //     text = text.replace(boldPattern, (match, content) => {
        //         return `<strong>${escapeHtml(content.trim())}</strong>`;
        //     });
        
        //     // Replace italic text
        //     text = text.replace(italicPattern, (match, content) => {
        //         return `<em>${escapeHtml(content.trim())}</em>`;
        //     });
        
        //     // Replace strikethrough text
        //     text = text.replace(strikethroughPattern, (match, content) => {
        //         return `<del>${escapeHtml(content.trim())}</del>`;
        //     });
        
        //     // Handle bullet points
        //     const bulletList = [];
        //     text = text.replace(bulletPointPattern, (match, content) => {
        //         bulletList.push(`<li>${escapeHtml(content.trim())}</li>`);
        //         return ""; // Remove from original text
        //     });

        //     // Wrap bullet points in unordered list
        //     if (bulletList.length) {
        //         text += `<ul>${bulletList.join("")}</ul>`;
        //     }
        
        //     // Handle numbered points
        //     const numberedList = [];
        //     text = text.replace(numberedPointPattern, (match, content) => {
        //         numberedList.push(`<li>${escapeHtml(content.trim())}</li>`);
        //         return ""; // Remove from original text
        //     });

        //     // Wrap numbered points in ordered list
        //     if (numberedList.length) {
        //         text += `<ol>${numberedList.join("")}</ol>`;
        //     }
        
        //     // Handle subpoints
        //     const subBulletList = [];
        //     text = text.replace(subBulletPointPattern, (match, content) => {
        //         subBulletList.push(`<li>${escapeHtml(content.trim())}</li>`);
        //         return ""; // Remove from original text
        //     });
        
        //     // Wrap subpoints in unordered list if there are any
        //     if (subBulletList.length) {
        //         text += `<ul style="list-style-type:circle;">${subBulletList.join("")}</ul>`;
        //     }
        
        //     // Handle blockquotes
        //     text = text.replace(blockquotePattern, (match, content) => {
        //         return `<blockquote>${escapeHtml(content.trim())}</blockquote>`;
        //     });
        
        //     // Handle horizontal rules
        //     text = text.replace(horizontalRulePattern, () => {
        //         return `<hr>`;
        //     });
        
        //     // Handle images
        //     text = text.replace(imagePattern, (match, altText, url) => {
        //         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}">`;
        //     });
        
        //     // Handle links
        //     text = text.replace(linkPattern, (match, linkText, url) => {
        //         return `<a href="${escapeHtml(url)}">${escapeHtml(linkText)}</a>`;
        //     });
        
        //     // Handle tables
        //     text = text.replace(tablePattern, (match, headerRow, _, rowData) => {
        //         const headerCells = headerRow.split('|').filter(cell => cell.trim() !== '');
        //         const rows = rowData.split('\n').map(row => {
        //             const cells = row.split('|').filter(cell => cell.trim() !== '');
        //             return `<tr>${cells.map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('')}</tr>`;
        //         });
        //         return `<table><thead><tr>${headerCells.map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('')}</tr></thead><tbody>$       {rows.join('')}</tbody></table>`;
        //     });
        
        //     // Handle task lists
        //     const taskList = [];
        //     text = text.replace(taskListPattern, (match, checked, content) => {
        //         taskList.push(`<li><input type="checkbox" ${checked === 'x' ? 'checked' : ''}> ${escapeHtml(content.trim())}</li>`);
        //         return ""; // Remove from original text
        //     });

        //     // Wrap task list in unordered list
        //     if (taskList.length) {
        //         text += `<ul>${taskList.join("")}</ul>`;
        //     }
        
        //     // Handle definition lists
        //     const definitionList = [];
        //     text = text.replace(definitionListPattern, (match, term, definition) => {
        //         definitionList.push(`<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`);
        //         return ""; // Remove from original text
        //     });

        //     // Wrap definition list in a definition list tag
        //     if (definitionList.length) {
        //         text += `<dl>${definitionList.join("")}</dl>`;
        //     }
        
        //     // Handle footnotes
        //     const footnotes = {};
        //     text = text.replace(footnotePattern, (match, id, content) => {
        //         footnotes[id] = escapeHtml(content.trim());
        //         return ""; // Remove from original text
        //     });

        //     // Add footnotes at the end
        //     let footnoteHtml = '';
        //     for (const [id, content] of Object.entries(footnotes)) {
        //         footnoteHtml += `<div class="footnote" id="fn-${id}">[^${id}]: ${content}</div>`;
        //     }

        //     text += footnoteHtml;
        
        //     // Add paragraph tags around text sections
        //     const paragraphs = text.split('\n\n').map(paragraph => {
        //         return `<p>${paragraph.trim()}</p>`;
        //     }).join('\n');
        
        //     return paragraphs.trim(); // Return the final processed text
        // }

        // // Example usage
        // const inputText = `
        // # Main Heading
        // This is a **bold** statement, and this is _italicized_. 
        // Here is a strikethrough ~~this text~~.

        // > This is a blockquote.

        // ---

        // ![Alt text](https://example.com/image.jpg)

        // [Link to example](https://example.com)

        // | Header 1 | Header 2 |
        // |----------|----------|
        // | Row 1 Col 1 | Row 1 Col 2 |
        // | Row 2 Col 1 | Row 2 Col 2 |

        // * [x] Completed task
        // * [ ] Incomplete task

        // Term 1
        // : Definition for term 1

        // Term 2
        // : Definition for term 2

        // Here























        // function escapeHtml(unsafe) {
        //     return unsafe;
        //         // .replace(/&/g, "&amp;")
        //         // .replace(/</g, "&lt;")
        //         // .replace(/>/g, "&gt;")
        //         // .replace(/"/g, "&quot;")
        //         // .replace(/'/g, "&#039;");
        // }
        
    //     function formatCode(text) {
    //         // Replace code blocks with <pre><code> tags
    //         const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
    //         const inlineCodePattern = /`([^`]+)`/g; // Single backticks
    //         // Function to normalize whitespace while preserving new lines
    //         const normalizeWhitespace = (str) => {
    //             return str
    //                 .split('\n') // Split into lines
    //                 .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
    //                 .join('\n'); // Join lines back together
    //         };
    //         // Replace block code first
    //         text = text.replace(blockCodePattern, (match, p1) => {
    //             // Replace a specific word in the block code
    //             const modifiedBlock = p1.replace(/<br>/g, '\n');
    //             return `<pre id="codesnap" class="all-pre"><code class="hljs">${escapeHtml(normalizeWhitespace(modifiedBlock.trim()))}</code></pre>`;
    //         });
        
    //         // Then replace inline code
    //         text = text.replace(inlineCodePattern, (match, p1) => {
    //             // Replace a specific word in the inline code
    //             const modifiedInline = p1.replace(/<br>/g, '\n');
    //             return `<code>${escapeHtml(normalizeWhitespace(modifiedBlock.trim()))}</code>`;
    //         });

    //         return text;
    //     }

    //     function displayDiv(inputField) {
    //   const divElement1 = document.getElementById("myDiv1");
    //   const divElement2 = document.getElementById("myDiv2");
    //   if (inputField.value != "") {
    //    divElement1.style.display = "none";
    //    divElement2.style.display = "block";
    //   } else {
    //    divElement1.style.display = "block";
    //    divElement2.style.display = "none";
    //   }}



        // function formatCode(text) {
        //     // Replace code blocks with <pre><code> tags
        //     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
        //     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
            
        //     // Function to normalize whitespace
        //     const normalizeWhitespace = (str) => {
        //         return str.replace(/\s+/g, ' ').trim(); // Replace multiple spaces with a single space and trim
        //     };
        
        //     // Replace block code first
        //     text = text.replace(blockCodePattern, (match, p1) => {
        //         // Remove <br> tags and normalize whitespace in block code
        //         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
        //         const formattedBlock = escapeHtml(modifiedBlock);
        //         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
        //     });
        
        //     // Then replace inline code
        //     text = text.replace(inlineCodePattern, (match, p1) => {
        //         // Remove <br> tags and normalize whitespace in inline code
        //         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
        //         const formattedInline = escapeHtml(modifiedInline);
        //         return `<code class="inline-code">${formattedInline}</code>`;
        //     });
        
        //     return text;
        // }























// -------------- currect it--------------------

        // function formatCode(text) {
        //    // Replace code blocks with <pre><code> tags
        //    const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
        //    const inlineCodePattern = /`([^`]+)`/g; // Single backticks
        
        //    // Function to normalize whitespace while preserving new lines
        //    const normalizeWhitespace = (str) => {
        //        return str
        //            .split('\n') // Split into lines
        //            .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
        //            .join('\n'); // Join lines back together
        //    };
       
        //    // Replace block code first
        //    text = text.replace(blockCodePattern, (match, p1) => {
        //        // Remove <br> tags and normalize whitespace in block code
        //        const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
        //        const formattedBlock = escapeHtml(modifiedBlock);
        //        return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
        //    });
       
        //    // Then replace inline code
        //    text = text.replace(inlineCodePattern, (match, p1) => {
        //        // Remove <br> tags and normalize whitespace in inline code
        //        const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
        //        const formattedInline = escapeHtml(modifiedInline);
        //        return `<code class="inline-code">${formattedInline}</code>`;
        //    });
       
        //    return text;
        // }
        





























        // function formatCode(text) {
        //     // Replace code blocks with <pre><code> tags
        //     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
        //     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
        //     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
        //     const listPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
        //     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
        //     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
        //     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
            
        //     // Function to normalize whitespace while preserving new lines
        //     const normalizeWhitespace = (str) => {
        //         return str
        //             .split('\n') // Split into lines
        //             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
        //             .join('\n'); // Join lines back together
        //     };
        
        //     // Replace block code first
        //     text = text.replace(blockCodePattern, (match, p1) => {
        //         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
        //         const formattedBlock = escapeHtml(modifiedBlock);
        //         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
        //     });
        
        //     // Then replace inline code
        //     text = text.replace(inlineCodePattern, (match, p1) => {
        //         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
        //         const formattedInline = escapeHtml(modifiedInline);
        //         return `<code class="inline-code">${formattedInline}</code>`;
        //     });
        
        //     // Replace headings
        //     text = text.replace(headingPattern, (match, hashes, content) => {
        //         const level = hashes.length; // Heading level
        //         return `<h${level}>${escapeHtml(content.trim())}</h${level}>`;
        //     });
        
        //     // Replace unordered lists
        //     text = text.replace(listPattern, (match) => {
        //         return `<li>${escapeHtml(match.trim())}</li>`;
        //     });
        
        //     // Wrap unordered list items in <ul>
        //     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');
            
        //     // Replace blockquotes
        //     text = text.replace(blockquotePattern, (match, p1) => {
        //         const content = escapeHtml(p1.replace(/^> /, '').trim());
        //         return `<blockquote>${content}</blockquote>`;
        //     });
        
        //     // Replace bold text
        //     text = text.replace(emphasisPattern, (match, p1, p2) => {
        //         return `<strong>${escapeHtml(p2)}</strong>`;
        //     });
        
        //     // Replace italics text
        //     text = text.replace(italicsPattern, (match, p1, p2) => {
        //         return `<em>${escapeHtml(p2)}</em>`;
        //     });
        
        //     return text;
        // }




































//         function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*\d+\.\s+(.+)$/gm; // Ordered lists
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}>${escapeHtml(content.trim())}</h${level}>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, content) => {
//         return `<li>${escapeHtml(content.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     return text;
// }




























// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*\d+\.\s+(.+)$/gm; // Ordered lists
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}>${escapeHtml(content.trim())}</h${level}>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, content) => {
//         return `<li>${escapeHtml(content.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     return text;
// }
































// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*\d+\.\s+(.+)$/gm; // Ordered lists
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings with bold text
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}><strong>${escapeHtml(content.trim())}</strong></h${level}>`;
//     });

//     // Replace unordered lists with nesting support
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace ordered lists with nesting support
//     text = text.replace(orderedListPattern, (match, content) => {
//         return `<li>${escapeHtml(content.trim())}</li>`;
//     });
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     return text;
// }




























































































// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const nestedListPattern = /^\s*(\d+)\.\s+(.+)$/gm; // Nested lists
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold or Underline
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const highlightPattern = /==(.+?)==/g; // Highlight
//     const definitionListPattern = /^([^\n:]+):\s*(.*)$/gm; // Definition lists
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]: (.+)/g; // Footnotes with definitions
//     const footnoteReferencePattern = /\[\^(\d+)\]/g; // Footnote references
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const escapedCharPattern = /\\([`*_{}[\]()#+\-!.>])/g; // Escaped characters
//     const superscriptPattern = /\^(\w+)/g; // Superscript
//     const subscriptPattern = /_(\w+)/g; // Subscript
//     const inlineHtmlPattern = /<([^>]+)>/g; // Inline HTML
//     const coloredTextPattern = /\{color:(.+?)\}(.+?)\{\/color\}/g; // Colored text

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}><strong>${escapeHtml(content.trim())}</strong></h${level}>`;
//     });

//     // Replace ordered lists with bold and larger points
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Handle nested lists
//     text = text.replace(nestedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold or underlined text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace underlined text
//     text = text.replace(/__(.*?)__/g, (match, p1) => {
//         return `<u>${escapeHtml(p1)}</u>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace highlighted text
//     text = text.replace(highlightPattern, (match, p1) => {
//         return `<mark>${escapeHtml(p1)}</mark>`;
//     });

//     // Replace colored text
//     text = text.replace(coloredTextPattern, (match, color, content) => {
//         return `<span style="color:${escapeHtml(color)};">${escapeHtml(content)}</span>`;
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (with definitions)
//     text = text.replace(footnotePattern, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(footnoteReferencePattern, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(escapedCharPattern, '$1');

//     // Handle superscripts
//     text = text.replace(superscriptPattern, (match, content) => {
//         return `<sup>${escapeHtml(content)}</sup>`;
//     });

//     // Handle subscripts
//     text = text.replace(subscriptPattern, (match, content) => {
//         return `<sub>${escapeHtml(content)}</sub>`;
//     });

//     // Handle inline HTML
//     text = text.replace(inlineHtmlPattern, (match, content) => {
//         return `<span>${content}</span>`; // Allow inline HTML
//     });

//     return text;
// }

// // Dummy escapeHtml function to sanitize HTML input
// function escapeHtml(unsafe) {
//     return unsafe
//         .replace(/&/g, "&amp;")
//         .replace(/</g, "&lt;")
//         .replace(/>/g, "&gt;")
//         .replace(/"/g, "&quot;")
//         .replace(/'/g, "&#039;");
// }









































































// function escapeHtml(unsafe) {
//     const element = document.createElement('div');
//     element.innerText = unsafe;
//     return element.innerHTML;
// }

// function formatCode(text) {
//     // Patterns for various markdown elements
//     const patterns = {
//         blockCode: /```(\w+)?\s*([\s\S]*?)```/g, // Triple backticks with optional language
//         inlineCode: /`([^`]+)`/g, // Single backticks
//         unorderedList: /^\s*([-*+]\s+.+)$/gm, // Unordered lists
//         orderedList: /^\s*(\d+)\.\s+(.+?)$/gm, // Ordered lists
//         customList: /^\s*=>\s+(.+)$/gm, // Custom lists
//         blockquote: /^(> .+?)(?:\s+\{cite:(.+?)\})?(?:\s+-\s+(.+))?$/gm, // Blockquotes with citation
//         emphasis: /(\*\*|__)(.*?)\1/g, // Bold or Underline
//         italics: /(\*|_)(.*?)\1/g, // Italics
//         strikethrough: /~~([\s\S]*?)~~|--([\s\S]*?)--/g, // Strikethrough
//         inlineBlock: /\{block:([^\}]+)\}(.*?)\{\/block\}/g, // Inline block elements
//         highlight: /\{highlight:([^\}]+)\}(.*?)\{\/highlight\}/g, // Highlight with color
//         definitionList: /^([^\n:]+):\s*(.*)$/gm, // Definition lists
//         link: /\[([^\]]+)\]\(([^)]+)\)/g, // Links
//         image: /!\[([^\]]*)\]\(([^)]+)(?: "([^"]*)")?\)/g, // Images with optional title
//         horizontalRule: /(^[-*]{3,}|^_{3,}|^~{3,})/gm, // Horizontal rules
//         taskList: /^\s*([-*+]\s+\[(x| )\]\s+.+)$/gm, // Task lists
//         footnote: /\[\^(\d+)\]:\s*(.+)/g, // Footnotes
//         footnoteReference: /\[\^(\d+)\]/g, // Footnote references
//         table: /(\|.*?\|(\n|$))/g, // Tables
//         escapedChar: /\\([`*_{}[\]()#+\-!.>])/g, // Escaped characters
//         superscript: /\^(\w+)/g, // Superscript
//         subscript: /_(\w+)/g, // Subscript
//         inlineHtml: /<([^>]+)>/g, // Inline HTML
//         paragraphBreak: /(\n\s*\n)/g, // Paragraph breaks
//         mathExpression: /\$(.+?)\$/g, // Math expressions
//     };

//     // Replace block code with language support
//     text = text.replace(patterns.blockCode, (match, language, codeBlock) => {
//         const formattedBlock = escapeHtml(codeBlock.trim().replace(/&lt;br&gt;/g, '\n'));
//         const langClass = language ? ` class="language-${escapeHtml(language)}"` : '';
//         return `<pre${langClass}><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(patterns.inlineCode, (match, code) => {
//         const formattedInline = escapeHtml(code.trim());
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace inline block elements
//     text = text.replace(patterns.inlineBlock, (match, className, content) => {
//         return `<div class="${escapeHtml(className)}">${escapeHtml(content.trim())}</div>`;
//     });

//     // Replace highlighted text with color
//     text = text.replace(patterns.highlight, (match, color, content) => {
//         return `<mark style="background-color:${escapeHtml(color)};">${escapeHtml(content.trim())}</mark>`;
//     });

//     // Replace custom lists
//     text = text.replace(patterns.customList, (match, content) => {
//         return `<li>${escapeHtml(content.trim())}</li>`;
//     });

//     // Replace ordered lists
//     text = text.replace(patterns.orderedList, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Replace unordered lists
//     text = text.replace(patterns.unorderedList, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes with optional citation
//     text = text.replace(patterns.blockquote, (match, quote, citation, author) => {
//         const content = escapeHtml(quote.replace(/^> /, '').trim());
//         const citationText = citation ? ` <cite>${escapeHtml(citation)}</cite>` : '';
//         return `<blockquote>${content}${citationText}${author ? `<footer>${escapeHtml(author.trim())}</footer>` : ''}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(patterns.emphasis, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2.trim())}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(patterns.italics, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2.trim())}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(patterns.strikethrough, (match, p1, p2) => {
//         return `<del>${escapeHtml(p1 || p2).trim()}</del>`;
//     });

//     // Replace math expressions
//     text = text.replace(patterns.mathExpression, (match, expression) => {
//         return `<span class="math">${escapeHtml(expression.trim())}</span>`;
//     });

//     // Replace definition lists
//     text = text.replace(patterns.definitionList, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(patterns.link, (match, linkText, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(linkText.trim())}</a>`;
//     });

//     // Replace images with optional title
//     text = text.replace(patterns.image, (match, altText, url, title) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}"${title ? ` title="${escapeHtml(title)}"` : ''} />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(patterns.horizontalRule, '<hr />');

//     // Replace task lists
//     text = text.replace(patterns.taskList, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes
//     text = text.replace(patterns.footnote, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(patterns.footnoteReference, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(patterns.table, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const headerRow = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const separatorRow = rows[1] ? rows[1].split('|').slice(1, -1) : [];
//         const hasSeparator = separatorRow.every(cell => cell.trim().match(/-+/));

//         const body = hasSeparator ? rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('') : '';

//         return `<table><thead><tr>${headerRow}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(patterns.escapedChar, (match, char) => {
//         return escapeHtml(char);
//     });

//     // Handle superscript
//     text = text.replace(patterns.superscript, (match, text) => {
//         return `<sup>${escapeHtml(text)}</sup>`;
//     });

//     // Handle subscript
//     text = text.replace(patterns.subscript, (match, text) => {
//         return `<sub>${escapeHtml(text)}</sub>`;
//     });

//     // Allow inline HTML
//     text = text.replace(patterns.inlineHtml, (match) => {
//         return match; // Directly include inline HTML
//     });

//     // Replace paragraph breaks with <p>
//     text = text.replace(patterns.paragraphBreak, '<p></p>');

//     return text;
// }

// // // Example usage
// // const markdownText = `
// // Here is some inline code: \`console.log("Hello, world!");\`

// // And here is a code block:
// // \`\`\`javascript
// // console.log("Hello, world!");
// // \`\`\`

// // Here is a list:
// // - Item 1
// // - Item 2

// // > This is a blockquote.
// // > {cite: Author Name}
// // `;

// // const formattedHtml = formatCode(markdownText);
// // console.log(formattedHtml);








































// function escapeHtml(str) {
//     const element = document.createElement('div');
//     element.innerText = str;
//     return element.innerHTML;
// }
// // Dummy escapeHtml function to sanitize HTML input
// function escapeHtml(unsafe) {
//     return unsafe;
// }
// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```(\w+)?\s*([\s\S]*?)```/g; // Triple backticks with optional language
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const customListPattern = /^\s*=>\s+(.+)$/gm; // Custom lists
//     const blockquotePattern = /^(> .+?)(?:\s+\{cite:(.+?)\})?(?:\s+-\s+(.+))?$/gm; // Blockquotes with citation
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold or Underline
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~([\s\S]*?)~~|--([\s\S]*?)--/g; // Strikethrough (multi-line support)
//     const inlineBlockPattern = /\{block:([^\}]+)\}(.*?)\{\/block\}/g; // Inline block elements
//     const highlightPattern = /\{highlight:([^\}]+)\}(.*?)\{\/highlight\}/g; // Highlight with color
//     const definitionListPattern = /^([^\n:]+):\s*(.*)$/gm; // Definition lists
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)(?: "([^"]*)")?\)/g; // Images with optional title
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^~{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[(x| )\]\s+.+)$/gm; // Task lists with custom markers
//     const footnotePattern = /\[\^(\d+)\]:\s*(.+)/g; // Footnotes with definitions
//     const footnoteReferencePattern = /\[\^(\d+)\]/g; // Footnote references
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const escapedCharPattern = /\\([`*_{}[\]()#+\-!.>])/g; // Escaped characters
//     const superscriptPattern = /\^(\w+)/g; // Superscript
//     const subscriptPattern = /_(\w+)/g; // Subscript
//     const inlineHtmlPattern = /<([^>]+)>/g; // Inline HTML
//     const paragraphBreakPattern = /(\n\s*\n)/g; // Paragraph breaks
//     const mathExpressionPattern = /\$(.+?)\$/g; // Math expressions

//     // Replace block code with language support
//     text = text.replace(blockCodePattern, (match, language, p1) => {
//         const formattedBlock = escapeHtml(p1.replace(/<br>/g, '\n').trim());
//         const langClass = language ? ` class="language-${escapeHtml(language)}"` : '';
//         return `<pre${langClass}><span class="language">${escapeHtml(language || 'Code')}</span><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const formattedInline = escapeHtml(p1.replace(/<br>/g, '').trim());
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace inline block elements
//     text = text.replace(inlineBlockPattern, (match, className, content) => {
//         return `<div class="${escapeHtml(className)}">${escapeHtml(content.trim())}</div>`;
//     });

//     // Replace highlighted text with color
//     text = text.replace(highlightPattern, (match, color, content) => {
//         return `<mark style="background-color:${escapeHtml(color)};">${escapeHtml(content.trim())}</mark>`;
//     });

//     // Replace custom lists
//     text = text.replace(customListPattern, (match, content) => {
//         return `<li>${escapeHtml(content.trim())}</li>`;
//     });

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes with optional citation
//     text = text.replace(blockquotePattern, (match, quote, citation, author) => {
//         const content = escapeHtml(quote.replace(/^> /, '').trim());
//         const citationText = citation ? ` <cite>${escapeHtml(citation)}</cite>` : '';
//         return `<blockquote>${content}${citationText}${author ? `<footer>${escapeHtml(author.trim())}</footer>` : ''}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2.trim())}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2.trim())}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1, p2) => {
//         return `<del>${escapeHtml(p1 || p2).trim()}</del>`;
//     });

//     // Replace math expressions
//     text = text.replace(mathExpressionPattern, (match, expression) => {
//         return `<span class="math">${escapeHtml(expression.trim())}</span>`;
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text.trim())}</a>`;
//     });

//     // Replace images with optional title
//     text = text.replace(imagePattern, (match, altText, url, title) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}"${title ? ` title="${escapeHtml(title)}"` : ''} />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes
//     text = text.replace(footnotePattern, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(footnoteReferencePattern, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
        
//         // Split the first row for headers
//         const headerRow = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
        
//         // Check for a separator row (typically with hyphens)
//         const separatorRow = rows[1].split('|').slice(1, -1);
//         const hasSeparator = separatorRow.every(cell => cell.trim().match(/-+/)); // check if the row has only dashes
        
//         // Process body rows
//         const body = hasSeparator ? rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('') : '';

//         return `<table><thead><tr>${headerRow}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(escapedCharPattern, (match, char) => {
//         return escapeHtml(char);
//     });

//     // Handle superscript
//     text = text.replace(superscriptPattern, (match, text) => {
//         return `<sup>${escapeHtml(text)}</sup>`;
//     });

//     // Handle subscript
//     text = text.replace(subscriptPattern, (match, text) => {
//         return `<sub>${escapeHtml(text)}</sub>`;
//     });

//     // Allow inline HTML
//     text = text.replace(inlineHtmlPattern, (match) => {
//         return match; // Directly include inline HTML
//     });

//     // Replace paragraph breaks with <p>
//     text = text.replace(paragraphBreakPattern, '<p></p>');

//     return text;
// }

















// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```(\w+)?\s*([\s\S]*?)```/g; // Triple backticks with optional language
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const customListPattern = /^\s*=>\s+(.+)$/gm; // Custom lists
//     const blockquotePattern = /^(> .+?)(?:\s+\{style:(.+?)\})?(?:\s+-\s+(.+))?$/gm; // Blockquotes with style and attribution
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold or Underline
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~([\s\S]*?)~~/g; // Strikethrough (multi-line support)
//     const inlineBlockPattern = /\{block:([^\}]+)\}(.*?)\{\/block\}/g; // Inline block elements
//     const highlightPattern = /\{highlight:([^\}]+)\}(.*?)\{\/highlight\}/g; // Highlight with color
//     const definitionListPattern = /^([^\n:]+):\s*(.*)$/gm; // Definition lists
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)(?: "([^"]*)")?\)/g; // Images with optional title
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^~{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]: (.+)/g; // Footnotes with definitions
//     const footnoteReferencePattern = /\[\^(\d+)\]/g; // Footnote references
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const escapedCharPattern = /\\([`*_{}[\]()#+\-!.>])/g; // Escaped characters
//     const superscriptPattern = /\^(\w+)/g; // Superscript
//     const subscriptPattern = /_(\w+)/g; // Subscript
//     const inlineHtmlPattern = /<([^>]+)>/g; // Inline HTML
//     const paragraphBreakPattern = /(\n\s*\n)/g; // Paragraph breaks
//     const mathExpressionPattern = /\$(.+?)\$/g; // Math expressions

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n')
//             .map(line => line.replace(/\s+/g, ' ').trim())
//             .join('\n');
//     };

//     // Replace block code with language support
//     text = text.replace(blockCodePattern, (match, language, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         const langClass = language ? ` class="language-${escapeHtml(language)}"` : '';
//         return `<pre${langClass}><span class="language">${escapeHtml(language || 'Code')}</span><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace inline block elements
//     text = text.replace(inlineBlockPattern, (match, className, content) => {
//         return `<div class="${escapeHtml(className)}">${escapeHtml(content)}</div>`;
//     });

//     // Replace highlighted text with color
//     text = text.replace(highlightPattern, (match, color, content) => {
//         return `<mark style="background-color:${escapeHtml(color)};">${escapeHtml(content)}</mark>`;
//     });

//     // Replace custom lists
//     text = text.replace(customListPattern, (match, content) => {
//         return `<li>${escapeHtml(content)}</li>`;
//     });

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes with optional style and attribution
//     text = text.replace(blockquotePattern, (match, quote, style, author) => {
//         const content = escapeHtml(quote.replace(/^> /, '').trim());
//         const styleAttr = style ? ` class="${escapeHtml(style)}"` : '';
//         return `<blockquote${styleAttr}>${content}${author ? `<footer>${escapeHtml(author.trim())}</footer>` : ''}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace math expressions
//     text = text.replace(mathExpressionPattern, (match, expression) => {
//         return `<span class="math">${escapeHtml(expression)}</span>`;
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images with optional title
//     text = text.replace(imagePattern, (match, altText, url, title) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}"${title ? ` title="${escapeHtml(title)}"` : ''} />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes
//     text = text.replace(footnotePattern, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(footnoteReferencePattern, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(escapedCharPattern, (match, char) => {
//         return escapeHtml(char);
//     });

//     // Handle superscript
//     text = text.replace(superscriptPattern, (match, text) => {
//         return `<sup>${escapeHtml(text)}</sup>`;
//     });

//     // Handle subscript
//     text = text.replace(subscriptPattern, (match, text) => {
//         return `<sub>${escapeHtml(text)}</sub>`;
//     });

//     // Allow inline HTML
//     text = text.replace(inlineHtmlPattern, (match) => {
//         return match; // Directly include inline HTML
//     });

//     // Replace paragraph breaks with <p>
//     text = text.replace(paragraphBreakPattern, '<p></p>');

//     return text;
// }

// Dummy escapeHtml function to sanitize HTML input
// function escapeHtml(unsafe) {
//     return unsafe;
        // .replace(/&/g, "&amp;")
        // .replace(/</g, "&lt;")
        // .replace(/>/g, "&gt;")
        // .replace(/"/g, "&quot;")
        // .replace(/'/g, "&#039;");
// }


















































// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```(\{line-numbers\})?\s*([\s\S]*?)```/g; // Triple backticks with optional line numbers
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const customListPattern = /^\s*=>\s+(.+)$/gm; // Custom lists
//     const blockquotePattern = /^(> .+?)(?:\s+\{style:(.+?)\})?(?:\s+-\s+(.+))?$/gm; // Blockquotes with style and attribution
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold or Underline
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~([\s\S]*?)~~/g; // Strikethrough (multi-line support)
//     const inlineBlockPattern = /\{block:([^\}]+)\}(.*?)\{\/block\}/g; // Inline block elements
//     const highlightPattern = /\{highlight:([^\}]+)\}(.*?)\{\/highlight\}/g; // Highlight with color
//     const definitionListPattern = /^([^\n:]+):\s*(.*)$/gm; // Definition lists
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)(?: "([^"]*)")?\)/g; // Images with optional title
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^~{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]: (.+)/g; // Footnotes with definitions
//     const footnoteReferencePattern = /\[\^(\d+)\]/g; // Footnote references
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const escapedCharPattern = /\\([`*_{}[\]()#+\-!.>])/g; // Escaped characters
//     const superscriptPattern = /\^(\w+)/g; // Superscript
//     const subscriptPattern = /_(\w+)/g; // Subscript
//     const inlineHtmlPattern = /<([^>]+)>/g; // Inline HTML
//     const paragraphBreakPattern = /(\n\s*\n)/g; // Paragraph breaks
//     const mathExpressionPattern = /\$(.+?)\$/g; // Math expressions

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n')
//             .map(line => line.replace(/\s+/g, ' ').trim())
//             .join('\n');
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, lineNumbers, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre${lineNumbers ? ' class="line-numbers"' : ''}><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace inline block elements
//     text = text.replace(inlineBlockPattern, (match, className, content) => {
//         return `<div class="${escapeHtml(className)}">${escapeHtml(content)}</div>`;
//     });

//     // Replace highlighted text with color
//     text = text.replace(highlightPattern, (match, color, content) => {
//         return `<mark style="background-color:${escapeHtml(color)};">${escapeHtml(content)}</mark>`;
//     });

//     // Replace custom lists
//     text = text.replace(customListPattern, (match, content) => {
//         return `<li>${escapeHtml(content)}</li>`;
//     });

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes with optional style and attribution
//     text = text.replace(blockquotePattern, (match, quote, style, author) => {
//         const content = escapeHtml(quote.replace(/^> /, '').trim());
//         const styleAttr = style ? ` class="${escapeHtml(style)}"` : '';
//         return `<blockquote${styleAttr}>${content}${author ? `<footer>${escapeHtml(author.trim())}</footer>` : ''}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace math expressions
//     text = text.replace(mathExpressionPattern, (match, expression) => {
//         return `<span class="math">${escapeHtml(expression)}</span>`;
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images with optional title
//     text = text.replace(imagePattern, (match, altText, url, title) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}"${title ? ` title="${escapeHtml(title)}"` : ''} />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes
//     text = text.replace(footnotePattern, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(footnoteReferencePattern, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(escapedCharPattern, (match, char) => {
//         return escapeHtml(char);
//     });

//     // Handle superscript
//     text = text.replace(superscriptPattern, (match, text) => {
//         return `<sup>${escapeHtml(text)}</sup>`;
//     });

//     // Handle subscript
//     text = text.replace(subscriptPattern, (match, text) => {
//         return `<sub>${escapeHtml(text)}</sub>`;
//     });

//     // Allow inline HTML
//     text = text.replace(inlineHtmlPattern, (match) => {
//         return match; // Directly include inline HTML
//     });

//     // Replace paragraph breaks with <p>
//     text = text.replace(paragraphBreakPattern, '<p></p>');

//     return text;
// }
































// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```(\{line-numbers\})?\s*([\s\S]*?)```/g; // Triple backticks with optional line numbers
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const blockquotePattern = /^(> .+?)(?:\s+-\s+(.+))?$/gm; // Blockquotes with attribution
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold or Underline
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const colorStrikethroughPattern = /~~\{color:(.+?)\}(.*?)\{\/color\}~~/g; // Colored strikethrough
//     const highlightPattern = /==(.+?)==/g; // Highlight
//     const definitionListPattern = /^([^\n:]+):\s*(.*)$/gm; // Definition lists
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)(?: "([^"]*)")?\)/g; // Images with optional title
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^~{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]: (.+)/g; // Footnotes with definitions
//     const footnoteReferencePattern = /\[\^(\d+)\]/g; // Footnote references
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const escapedCharPattern = /\\([`*_{}[\]()#+\-!.>])/g; // Escaped characters
//     const superscriptPattern = /\^(\w+)/g; // Superscript
//     const subscriptPattern = /_(\w+)/g; // Subscript
//     const inlineHtmlPattern = /<([^>]+)>/g; // Inline HTML
//     const coloredTextPattern = /\{color:(.+?)\}(.+?)\{\/color\}/g; // Colored text
//     const sizeTextPattern = /\{size:(.+?)\}(.+?)\{\/size\}/g; // Custom font sizes
//     const bulletIconPattern = /^\s*([-*+]\s+)(🔹\s+)?(.+)$/gm; // Bullet points with icons
//     const paragraphBreakPattern = /(\n\s*\n)/g; // Paragraph breaks

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n')
//             .map(line => line.replace(/\s+/g, ' ').trim())
//             .join('\n');
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, lineNumbers, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre${lineNumbers ? ' class="line-numbers"' : ''}><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace colored text
//     text = text.replace(coloredTextPattern, (match, color, content) => {
//         return `<span style="color:${escapeHtml(color)};">${escapeHtml(content)}</span>`;
//     });

//     // Replace custom font sizes
//     text = text.replace(sizeTextPattern, (match, size, content) => {
//         return `<span style="font-size:${escapeHtml(size)};">${escapeHtml(content)}</span>`;
//     });

//     // Replace bullet points with icons
//     text = text.replace(bulletIconPattern, (match, bullet, icon, content) => {
//         return `<li>${icon || ''}${escapeHtml(content)}</li>`;
//     });

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes with attribution
//     text = text.replace(blockquotePattern, (match, quote, author) => {
//         const content = escapeHtml(quote.replace(/^> /, '').trim());
//         return `<blockquote>${content}${author ? `<footer>${escapeHtml(author.trim())}</footer>` : ''}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace colored strikethrough text
//     text = text.replace(colorStrikethroughPattern, (match, color, content) => {
//         return `<del style="color:${escapeHtml(color)};">${escapeHtml(content)}</del>`;
//     });

//     // Replace highlighted text
//     text = text.replace(highlightPattern, (match, p1) => {
//         return `<mark>${escapeHtml(p1)}</mark>`;
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images with optional title
//     text = text.replace(imagePattern, (match, altText, url, title) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}"${title ? ` title="${escapeHtml(title)}"` : ''} />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes
//     text = text.replace(footnotePattern, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(footnoteReferencePattern, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(escapedCharPattern, (match, char) => {
//         return escapeHtml(char);
//     });

//     // Handle superscript
//     text = text.replace(superscriptPattern, (match, text) => {
//         return `<sup>${escapeHtml(text)}</sup>`;
//     });

//     // Handle subscript
//     text = text.replace(subscriptPattern, (match, text) => {
//         return `<sub>${escapeHtml(text)}</sub>`;
//     });

//     // Allow inline HTML
//     text = text.replace(inlineHtmlPattern, (match) => {
//         return match; // Directly include inline HTML
//     });

//     // Replace paragraph breaks with <p>
//     text = text.replace(paragraphBreakPattern, '<p></p>');

//     return text;
// }






























// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold or Underline
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const highlightPattern = /==(.+?)==/g; // Highlight
//     const definitionListPattern = /^([^\n:]+):\s*(.*)$/gm; // Definition lists
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^~{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]: (.+)/g; // Footnotes with definitions
//     const footnoteReferencePattern = /\[\^(\d+)\]/g; // Footnote references
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const escapedCharPattern = /\\([`*_{}[\]()#+\-!.>])/g; // Escaped characters
//     const superscriptPattern = /\^(\w+)/g; // Superscript
//     const subscriptPattern = /_(\w+)/g; // Subscript
//     const inlineHtmlPattern = /<([^>]+)>/g; // Inline HTML
//     const coloredTextPattern = /\{color:(.+?)\}(.+?)\{\/color\}/g; // Colored text
//     const paragraphBreakPattern = /(\n\s*\n)/g; // Paragraph breaks

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace colored text
//     text = text.replace(coloredTextPattern, (match, color, content) => {
//         return `<span style="color:${escapeHtml(color)};">${escapeHtml(content)}</span>`;
//     });

//     // Replace ordered lists with bold and larger points
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace highlighted text
//     text = text.replace(highlightPattern, (match, p1) => {
//         return `<mark>${escapeHtml(p1)}</mark>`;
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (with definitions)
//     text = text.replace(footnotePattern, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(footnoteReferencePattern, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(escapedCharPattern, (match, char) => {
//         return escapeHtml(char);
//     });

//     // Handle superscript
//     text = text.replace(superscriptPattern, (match, text) => {
//         return `<sup>${escapeHtml(text)}</sup>`;
//     });

//     // Handle subscript
//     text = text.replace(subscriptPattern, (match, text) => {
//         return `<sub>${escapeHtml(text)}</sub>`;
//     });

//     // Allow inline HTML
//     text = text.replace(inlineHtmlPattern, (match) => {
//         return match; // Directly include inline HTML
//     });

//     // Replace paragraph breaks
//     text = text.replace(paragraphBreakPattern, '<p></p>');

//     return text;
// }



















// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const nestedListPattern = /^\s*(\d+)\.\s+(.+)$/gm; // Nested lists
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const highlightPattern = /==(.+?)==/g; // Highlight
//     const definitionListPattern = /^([^\n:]+):\s*(.*)$/gm; // Definition lists
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]: (.+)/g; // Footnotes with definitions
//     const footnoteReferencePattern = /\[\^(\d+)\]/g; // Footnote references
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const escapedCharPattern = /\\([`*_{}[\]()#+\-!.>])/g; // Escaped characters
//     const superscriptPattern = /\^(\w+)/g; // Superscript
//     const subscriptPattern = /_(\w+)/g; // Subscript
//     const inlineHtmlPattern = /<([^>]+)>/g; // Inline HTML

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });


//     // Replace ordered lists with bold and larger points
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Handle nested lists
//     text = text.replace(nestedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace highlighted text
//     text = text.replace(highlightPattern, (match, p1) => {
//         return `<mark>${escapeHtml(p1)}</mark>`;
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (with definitions)
//     text = text.replace(footnotePattern, (match, footnoteId, definition) => {
//         return `<sup id="footnote-${footnoteId}">[${escapeHtml(footnoteId)}]</sup><span class="footnote-definition">${escapeHtml(definition.trim())}</span>`;
//     });

//     // Replace footnote references
//     text = text.replace(footnoteReferencePattern, (match, footnoteId) => {
//         return `<a href="#footnote-${footnoteId}">${escapeHtml(match)}</a>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Handle escaped characters
//     text = text.replace(escapedCharPattern, (match, char) => {
//         return escapeHtml(char);
//     });

//     // Handle superscript
//     text = text.replace(superscriptPattern, (match, text) => {
//         return `<sup>${escapeHtml(text)}</sup>`;
//     });

//     // Handle subscript
//     text = text.replace(subscriptPattern, (match, text) => {
//         return `<sub>${escapeHtml(text)}</sub>`;
//     });

//     // Allow inline HTML
//     text = text.replace(inlineHtmlPattern, (match) => {
//         return match; // Directly include inline HTML
//     });

//     return text;
// }































// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}><strong>${escapeHtml(content.trim())}</strong></h${level}>`;
//     });

//     // Replace ordered lists with bold and larger points
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li style="font-size: 1.2em;"><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     return text;
// }









































// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+?):/gm; // Ordered lists with colon
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}><strong>${escapeHtml(content.trim())}</strong></h${level}>`;
//     });

//     // Replace ordered lists with bold points
//     text = text.replace(orderedListPattern, (match, number, content) => {
//         return `<li><strong>${escapeHtml(content.trim())}</strong></li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     return text;
// }



















// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*\d+\.\s+(.+):/gm; // Ordered lists with bold pattern
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Helper function to handle nested lists
//     const handleLists = (text) => {
//         let listStack = [];
//         let currentListType = null;

//         return text.split('\n').map(line => {
//             const matchOrdered = line.match(/^\s*\d+\.\s+/);
//             const matchUnordered = line.match(/^\s*[-*+]\s+/);
//             const isListItem = matchOrdered || matchUnordered;

//             if (isListItem) {
//                 const listItem = escapeHtml(line.trim());
//                 if (matchOrdered) {
//                     if (currentListType !== 'ol') {
//                         currentListType = 'ol';
//                         listStack.push(`<ol>`);
//                     }
//                     return `<li><strong>${listItem.replace(/^\d+\.\s+/, '')}</strong></li>`;
//                 } else if (matchUnordered) {
//                     if (currentListType !== 'ul') {
//                         currentListType = 'ul';
//                         listStack.push(`<ul>`);
//                     }
//                     return `<li>${listItem}</li>`;
//                 }
//             } else {
//                 if (currentListType) {
//                     listStack.push(`</${currentListType}>`);
//                     currentListType = null;
//                 }
//                 return line; // Non-list line
//             }
//         }).concat(listStack.reverse()).join('\n');
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings with bold text
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}><strong>${escapeHtml(content.trim())}</strong></h${level}>`;
//     });

//     // Handle lists (unordered and ordered) with bold points
//     text = handleLists(text);

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     return text;
// }





















// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*(\d+)\.\s+(.+):/gm; // Ordered lists with colon
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Helper function to handle nested lists
//     const handleLists = (text) => {
//         let listStack = [];
//         let currentListType = null;

//         return text.split('\n').map(line => {
//             const matchOrdered = line.match(/^\s*\d+\.\s+/);
//             const matchUnordered = line.match(/^\s*[-*+]\s+/);
//             const isListItem = matchOrdered || matchUnordered;

//             if (isListItem) {
//                 const listItem = escapeHtml(line.trim());
//                 if (matchOrdered) {
//                     if (currentListType !== 'ol') {
//                         currentListType = 'ol';
//                         listStack.push(`<ol>`);
//                     }
//                     // Bold the content after the number
//                     return `<li><strong>${listItem.replace(/^\d+\.\s+/, '')}</strong></li>`;
//                 } else if (matchUnordered) {
//                     if (currentListType !== 'ul') {
//                         currentListType = 'ul';
//                         listStack.push(`<ul>`);
//                     }
//                     return `<li>${listItem}</li>`;
//                 }
//             } else {
//                 if (currentListType) {
//                     listStack.push(`</${currentListType}>`);
//                     currentListType = null;
//                 }
//                 return line; // Non-list line
//             }
//         }).concat(listStack.reverse()).join('\n');
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings with bold text
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}><strong>${escapeHtml(content.trim())}</strong></h${level}>`;
//     });

//     // Handle lists (unordered and ordered) with bold points
//     text = handleLists(text);

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     return text;
// }






















// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*\d+\.\s+(.+):/gm; // Ordered lists with bold pattern
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables

//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     // Helper function to handle nested lists
//     const handleLists = (text) => {
//         let listStack = [];
//         let currentListType = null;

//         return text.split('\n').map(line => {
//             const matchOrdered = line.match(/^\s*\d+\.\s+/);
//             const matchUnordered = line.match(/^\s*[-*+]\s+/);
//             const isListItem = matchOrdered || matchUnordered;

//             if (isListItem) {
//                 const listItem = escapeHtml(line.trim());
//                 if (matchOrdered) {
//                     if (currentListType !== 'ol') {
//                         currentListType = 'ol';
//                         listStack.push(`<ol>`);
//                     }
//                     return `<li><strong>${listItem.replace(/^\d+\.\s+/, '')}</strong></li>`;
//                 } else if (matchUnordered) {
//                     if (currentListType !== 'ul') {
//                         currentListType = 'ul';
//                         listStack.push(`<ul>`);
//                     }
//                     return `<li>${listItem}</li>`;
//                 }
//             } else {
//                 if (currentListType) {
//                     listStack.push(`</${currentListType}>`);
//                     currentListType = null;
//                 }
//                 return line; // Non-list line
//             }
//         }).concat(listStack.reverse()).join('\n');
//     };

//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings with bold text
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}><strong>${escapeHtml(content.trim())}</strong></h${level}>`;
//     });

//     // Handle lists (unordered and ordered) with bold points
//     text = handleLists(text);

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     return text;
// }






































//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

// function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*\d+\.\s+(.+)$/gm; // Ordered lists
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const linkWithTitlePattern = /\[([^\]]+)\]\(([^)]+) "([^"]+)"\)/g; // Links with title
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const htmlBlockPattern = /(<[a-z][\s\S]*?>[\s\S]*?<\/[a-z]>)/gi; // HTML block elements
//     const definitionListPattern = /^(.+?)\s*:\s*(.+)$/gm; // Definition lists
//     const abbreviationPattern = /\*\[([^\]]+)\]:\s*(.+)$/gm; // Abbreviations
//     const commentPattern = /<!--(.*?)-->/g; // HTML comments





//     // Replace headings
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}>${escapeHtml(content.trim())}</h${level}>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, content) => {
//         return `<li>${escapeHtml(content.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace links with title
//     text = text.replace(linkWithTitlePattern, (match, text, url, title) => {
//         return `<a href="${escapeHtml(url)}" title="${escapeHtml(title)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });
//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });
//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Replace HTML block elements
//     text = text.replace(htmlBlockPattern, (match) => {
//         return match; // Allow raw HTML elements
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dl><dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd></dl>`;
//     });

//     // Replace abbreviations
//     text = text.replace(abbreviationPattern, (match, term, definition) => {
//         return `<abbr title="${escapeHtml(definition.trim())}">${escapeHtml(term.trim())}</abbr>`;
//     });

//     // Remove comments (optional)
//     text = text.replace(commentPattern, '');

//     return text;
// }





























































































//     // Function to normalize whitespace while preserving new lines
//     const normalizeWhitespace = (str) => {
//         return str
//             .split('\n') // Split into lines
//             .map(line => line.replace(/\s+/g, ' ').trim()) // Normalize each line
//             .join('\n'); // Join lines back together
//     };

//     function formatCode(text) {
//     // Patterns for various markdown elements
//     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
//     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
//     const headingPattern = /^(#{1,6})\s*(.+)$/gm; // Headings
//     const unorderedListPattern = /^\s*([-*+]\s+.+)$/gm; // Unordered lists
//     const orderedListPattern = /^\s*\d+\.\s+(.+)$/gm; // Ordered lists
//     const blockquotePattern = /^(> .+)$/gm; // Blockquotes
//     const emphasisPattern = /(\*\*|__)(.*?)\1/g; // Bold
//     const italicsPattern = /(\*|_)(.*?)\1/g; // Italics
//     const strikethroughPattern = /~~(.*?)~~/g; // Strikethrough
//     const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g; // Links
//     const linkWithTitlePattern = /\[([^\]]+)\]\(([^)]+) "([^"]+)"\)/g; // Links with title
//     const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g; // Images
//     const horizontalRulePattern = /(^[-*]{3,}|^_{3,}|^\*{3,})/gm; // Horizontal rules
//     const taskListPattern = /^\s*([-*+]\s+\[([ x])\]\s+.+)$/gm; // Task lists
//     const footnotePattern = /\[\^(\d+)\]/g; // Footnotes
//     const tablePattern = /(\|.*?\|(\n|$))/g; // Tables
//     const htmlBlockPattern = /(<[a-z][\s\S]*?>[\s\S]*?<\/[a-z]>)/gi; // HTML block elements
//     const definitionListPattern = /^(.+?)\s*:\s*(.+)$/gm; // Definition lists
//     const abbreviationPattern = /\*\[([^\]]+)\]:\s*(.+)$/gm; // Abbreviations
//     const commentPattern = /<!--(.*?)-->/g; // HTML comments



//     // Replace block code first
//     text = text.replace(blockCodePattern, (match, p1) => {
//         const modifiedBlock = normalizeWhitespace(p1.replace(/<br>/g, '\n'));
//         const formattedBlock = escapeHtml(modifiedBlock);
//         return `<pre><code class="formatted-code">${formattedBlock}</code></pre>`;
//     });

//     // Replace inline code
//     text = text.replace(inlineCodePattern, (match, p1) => {
//         const modifiedInline = normalizeWhitespace(p1.replace(/<br>/g, ''));
//         const formattedInline = escapeHtml(modifiedInline);
//         return `<code class="inline-code">${formattedInline}</code>`;
//     });

//     // Replace headings
//     text = text.replace(headingPattern, (match, hashes, content) => {
//         const level = hashes.length; // Heading level
//         return `<h${level}>${escapeHtml(content.trim())}</h${level}>`;
//     });

//     // Replace unordered lists
//     text = text.replace(unorderedListPattern, (match) => {
//         return `<li>${escapeHtml(match.trim())}</li>`;
//     });

//     // Wrap unordered list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace ordered lists
//     text = text.replace(orderedListPattern, (match, content) => {
//         return `<li>${escapeHtml(content.trim())}</li>`;
//     });

//     // Wrap ordered list items in <ol>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ol>$1</ol>');

//     // Replace blockquotes
//     text = text.replace(blockquotePattern, (match, p1) => {
//         const content = escapeHtml(p1.replace(/^> /, '').trim());
//         return `<blockquote>${content}</blockquote>`;
//     });

//     // Replace bold text
//     text = text.replace(emphasisPattern, (match, p1, p2) => {
//         return `<strong>${escapeHtml(p2)}</strong>`;
//     });

//     // Replace italics text
//     text = text.replace(italicsPattern, (match, p1, p2) => {
//         return `<em>${escapeHtml(p2)}</em>`;
//     });

//     // Replace strikethrough text
//     text = text.replace(strikethroughPattern, (match, p1) => {
//         return `<del>${escapeHtml(p1)}</del>`;
//     });

//     // Replace links
//     text = text.replace(linkPattern, (match, text, url) => {
//         return `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace links with title
//     text = text.replace(linkWithTitlePattern, (match, text, url, title) => {
//         return `<a href="${escapeHtml(url)}" title="${escapeHtml(title)}">${escapeHtml(text)}</a>`;
//     });

//     // Replace images
//     text = text.replace(imagePattern, (match, altText, url) => {
//         return `<img src="${escapeHtml(url)}" alt="${escapeHtml(altText)}" />`;
//     });

//     // Replace horizontal rules
//     text = text.replace(horizontalRulePattern, '<hr />');

//     // Replace task lists
//     text = text.replace(taskListPattern, (match, listItem, checked) => {
//         const status = checked === 'x' ? 'checked' : '';
//         return `<li><input type="checkbox" ${status} /> ${escapeHtml(listItem.trim())}</li>`;
//     });

//     // Wrap task list items in <ul>
//     text = text.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');

//     // Replace footnotes (basic)
//     text = text.replace(footnotePattern, (match, footnoteId) => {
//         return `<sup>[${escapeHtml(footnoteId)}]</sup>`;
//     });

//     // Replace tables
//     text = text.replace(tablePattern, (match) => {
//         const rows = match.split('\n').filter(row => row.trim());
//         const header = rows[0].split('|').slice(1, -1).map(cell => `<th>${escapeHtml(cell.trim())}</th>`).join('');
//         const body = rows.slice(2).map(row => {
//             const cells = row.split('|').slice(1, -1).map(cell => `<td>${escapeHtml(cell.trim())}</td>`).join('');
//             return `<tr>${cells}</tr>`;
//         }).join('');
//         return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
//     });

//     // Replace HTML block elements
//     text = text.replace(htmlBlockPattern, (match) => {
//         return match; // Allow raw HTML elements
//     });

//     // Replace definition lists
//     text = text.replace(definitionListPattern, (match, term, definition) => {
//         return `<dl><dt>${escapeHtml(term.trim())}</dt><dd>${escapeHtml(definition.trim())}</dd></dl>`;
//     });

//     // Replace abbreviations
//     text = text.replace(abbreviationPattern, (match, term, definition) => {
//         return `<abbr title="${escapeHtml(definition.trim())}">${escapeHtml(term.trim())}</abbr>`;
//     });

//     // Remove comments (optional)
//     text = text.replace(commentPattern, '');

//     return text;
// }































        // function formatCode(text) {
        //     // Replace code blocks with <pre><code> tags
        //     const blockCodePattern = /```([\s\S]*?)```/g; // Triple backticks
        //     const inlineCodePattern = /`([^`]+)`/g; // Single backticks
            
        //     // Replace block code first
        //     text = text.replace(blockCodePattern, (match, p1) => {
        //         // Escape and trim block code first
        //         const escapedBlock = escapeHtml(p1.trim());
        //         // Replace a specific word in the escaped block code
        //         const modifiedBlock = escapedBlock.replace(/<br>/g, '\n');
        //         return `<pre><code>${modifiedBlock}</code></pre>`;
        //     });
        
        //     // Then replace inline code
        //     text = text.replace(inlineCodePattern, (match, p1) => {
        //         // Escape and trim inline code first
        //         const escapedInline = escapeHtml(p1.trim());
        //         // Replace a specific word in the escaped inline code
        //         const modifiedInline = escapedInline.replace(/<br>/g, '\n');
        //         return `<code>${modifiedInline}</code>`;
        //     });
            
        //     return text;
        // }





        
        // // Example usage
        // const inputText = `
        // Here is some inline code: \`const x = 10;\`
        
        // And here is a code block:
        
        // \`\`\`
        // function test() {
        //     console.log("Hello, World!");
        // }
        // \`\`\`
        // `;
        
        // console.log(formatCode(inputText));










// -------------- currect it--------------------

        // function escapeHtml(unsafe) {
        //     return unsafe
        //         .replace(/&/g, "&amp;")
        //         .replace(/</g, "&lt;")
        //         .replace(/>/g, "&gt;")
        //         .replace(/"/g, "&quot;")
        //         .replace(/'/g, "&#039;");
        // }

        // function formatCode(text) {
        //     // Regular expressions for different patterns
        //     const blockCodePattern = /```(\w+)?\s*([\s\S]*?)```/g; // Triple backticks with optional language
        //     const tildeBlockCodePattern = /~~~(\w+)?\s*([\s\S]*?)~~~/g; // Tilde for code block with optional language
        //     const inlineCodePattern = /`([^`]+)`/g; // Single backticks for inline code
        //     const headingPattern = /^(#+)\s*(.*)$/gm; // Headings (e.g., #, ##, ###)
        //     const bulletPointPattern = /^\s*\*\s+(.*)$/gm; // Bullet points (e.g., * item)
        //     const numberedPointPattern = /^\s*\d+\.\s+(.*)$/gm; // Numbered points (e.g., 1. item)
        
        //     // Replace headings
        //     text = text.replace(headingPattern, (match, hashes, content) => {
        //         const level = hashes.length; // Number of '#' determines heading level
        //         return `<h${level}>${escapeHtml(content.trim())}</h${level}>`;
        //     });
        
        //     // Wrap bullet points in an unordered list
        //     const bulletList = [];
        //     text = text.replace(bulletPointPattern, (match, content) => {
        //         bulletList.push(`<li>${escapeHtml(content.trim())}</li>`);
        //         return ""; // Remove from original text
        //     });
        //     if (bulletList.length) {
        //         text += `<ul>${bulletList.join("")}</ul>`;
        //     }
        
        //     // Wrap numbered points in an ordered list
        //     const numberedList = [];
        //     text = text.replace(numberedPointPattern, (match, content) => {
        //         numberedList.push(`<li>${escapeHtml(content.trim())}</li>`);
        //         return ""; // Remove from original text
        //     });
        //     if (numberedList.length) {
        //         text += `<ol>${numberedList.join("")}</ol>`;
        //     }
        
        //     // Replace block code first (supporting both backticks and tildes)
        //     text = text.replace(blockCodePattern, (match, lang, p1) => {
        //         const languageClass = lang ? ` class="${lang}"` : '';
        //         return `<pre${languageClass}><code>${escapeHtml(p1.trim())}</code></pre>`;
        //     });
        
        //     text = text.replace(tildeBlockCodePattern, (match, lang, p1) => {
        //         const languageClass = lang ? ` class="${lang}"` : '';
        //         return `<pre${languageClass}><code>${escapeHtml(p1.trim())}</code></pre>`;
        //     });
        
        //     // Then replace inline code
        //     text = text.replace(inlineCodePattern, (match, p1) => {
        //         return `<code>${escapeHtml(p1.trim())}</code>`;
        //     });
        
        //     return text;
        // }

        // // Example usage
        // const inputText = `
        // # Heading Level 1
        // Here is some inline code: \`const x = 10;\`

        // ## Heading Level 2
        // And here is a JavaScript code block:

        // \`\`\`javascript
        // function test() {
        //     console.log("Hello, World!");
        // }
        // \`\`\`

        // ### Heading Level 3
        // Here is a bullet list:
        // * Item 1
        //     * Sub-item 1.1
        //     * Sub-item 1.2
        // * Item 2

        // Here is a numbered list:
        // 1. First item
        // 2. Second item
        //    1. Sub-item 2.1
        //    2. Sub-item 2.2

        // Here is a Python code block:

        // ~~~python
        // def greet():
        //     print("Hello, World!")
        // ~~~
        // `;

        // console.log(formatCode(inputText));

        


















        // function escapeHtml(unsafe) {
        //     return unsafe
        //         .replace(/&/g, "&amp;")
        //         .replace(/</g, "&lt;")
        //         .replace(/>/g, "&gt;")
        //         .replace(/"/g, "&quot;")
        //         .replace(/'/g, "&#039;");
        // }

        // function formatCode(text) {
        //     // Regular expressions for different patterns
        //     const blockCodePattern = /```(\w+)?\s*([\s\S]*?)```/g; // Triple backticks with optional language
        //     const tildeBlockCodePattern = /~~~(\w+)?\s*([\s\S]*?)~~~/g; // Tilde for code block with optional language
        //     const inlineCodePattern = /`([^`]+)`/g; // Single backticks for inline code
        //     const headingPattern = /^(#+)\s*(.*)$/gm; // Headings (e.g., #, ##, ###)
        //     const bulletPointPattern = /^\s*\*\s+(.*)$/gm; // Bullet points (e.g., * item)
        //     const numberedPointPattern = /^\s*\d+\.\s+(.*)$/gm; // Numbered points (e.g., 1. item)
        //     const subBulletPointPattern = /^\s*\*\*\s+(.*)$/gm; // Subpoints (e.g., ** sub-item)
        
        //     // Replace headings
        //     text = text.replace(headingPattern, (match, hashes, content) => {
        //         const level = hashes.length; // Number of '#' determines heading level
        //         return `\n<h${level}>${escapeHtml(content.trim())}</h${level}>\n`;
        //     });
        
        //     // Replace bullet points
        //     const bulletList = [];
        //     text = text.replace(bulletPointPattern, (match, content) => {
        //         bulletList.push(`${escapeHtml(content.trim())}`);
        //         return ""; // Remove from original text
        //     });

        //     // Wrap bullet points in an unordered list
        //     if (bulletList.length) {
        //         text += `<ul>\n<li>${bulletList.join("</li>\n<li>")}</li>\n</ul>\n`;
        //     }
        
        //     // Replace numbered points
        //     const numberedList = [];
        //     text = text.replace(numberedPointPattern, (match, content) => {
        //         numberedList.push(`${escapeHtml(content.trim())}`);
        //         return ""; // Remove from original text
        //     });
        
        //     // Wrap numbered points in an ordered list
        //     if (numberedList.length) {
        //         text += `<ol>\n<li>${numberedList.join("</li>\n<li>")}</li>\n</ol>\n`;
        //     }
        
        //     // Replace block code first (supporting both backticks and tildes)
        //     text = text.replace(blockCodePattern, (match, lang, p1) => {
        //         const languageClass = lang ? ` class="${lang}"` : '';
        //         return `\n<pre${languageClass}>\n${escapeHtml((p1.replace(/<br>/g, '\n')).trim())}\n</pre>\n`;
        //     });
        
        //     text = text.replace(tildeBlockCodePattern, (match, lang, p1) => {
        //         const languageClass = lang ? ` class="${lang}"` : '';
        //         return `\n<pre${languageClass}>\n${escapeHtml((p1.replace(/<br>/g, '\n')).trim())}\n</pre>\n`;
        //     });
        
        //     // Then replace inline code
        //     text = text.replace(inlineCodePattern, (match, p1) => {
        //         return `<code>${escapeHtml((p1.replace(/<br>/g, '\n')).trim())}</code>`;
        //     });
        
        //     return text.trim(); // Return the final processed text
        // }

        // // Example usage
        // const inputText = `
        // # Heading Level 1
        // Here is some inline code: \`const x = 10;\`

        // ## Heading Level 2
        // And here is a JavaScript code block:

        // \`\`\`javascript
        // function test() {
        //     console.log("Hello, World!");
        // }
        // \`\`\`

        // ### Heading Level 3
        // Here is a bullet list:
        // * Item 1
        //     * Sub-item 1.1
        //     * Sub-item 1.2
        // * Item 2

        // Here is a numbered list:
        // 1. First item
        // 2. Second item
        //    1. Sub-item 2.1
        //    2. Sub-item 2.2

        // Here is a Python code block:

        // ~~~python
        // def greet():
        //     print("Hello, World!")
        // ~~~
        // `;

        // console.log(formatCode(inputText));





































        // function escapeHtml(unsafe) {
        //     return unsafe
        //         .replace(/&/g, "&amp;")
        //         .replace(/</g, "&lt;")
        //         .replace(/>/g, "&gt;")
        //         .replace(/"/g, "&quot;")
        //         .replace(/'/g, "&#039;");
        // }

        // function formatCode(text) {
        //     // Regular expressions for different patterns
        //     const blockCodePattern = /```(\w+)?\s*([\s\S]*?)```/g; // Triple backticks with optional language
        //     const tildeBlockCodePattern = /~~~(\w+)?\s*([\s\S]*?)~~~/g; // Tilde for code block with optional language
        //     const inlineCodePattern = /`([^`]+)`/g; // Single backticks for inline code
        //     const headingPattern = /^(#+)\s*(.*)$/gm; // Headings (e.g., #, ##, ###)
        //     const bulletPointPattern = /^\s*\*\s+(.*)$/gm; // Bullet points (e.g., * item)
        //     const numberedPointPattern = /^\s*\d+\.\s+(.*)$/gm; // Numbered points (e.g., 1. item)
        
        //     // Replace headings
        //     text = text.replace(headingPattern, (match, hashes, content) => {
        //         const level = hashes.length; // Number of '#' determines heading level
        //         return `\n<h${level}>${escapeHtml(content.trim())}</h${level}>\n`;
        //     });
        
        //     // Replace bullet points
        //     const bulletList = [];
        //     text = text.replace(bulletPointPattern, (match, content) => {
        //         bulletList.push(`${escapeHtml(content.trim())}`);
        //         return ""; // Remove from original text
        //     });

        //     // Wrap bullet points in an unordered list
        //     if (bulletList.length) {
        //         text += `<ul>\n<li>${bulletList.join("</li>\n<li>")}</li>\n</ul>\n`;
        //     }
        
        //     // Replace numbered points
        //     const numberedList = [];
        //     text = text.replace(numberedPointPattern, (match, content) => {
        //         numberedList.push(`${escapeHtml(content.trim())}`);
        //         return ""; // Remove from original text
        //     });
        
        //     // Wrap numbered points in an ordered list
        //     if (numberedList.length) {
        //         text += `<ol>\n<li>${numberedList.join("</li>\n<li>")}</li>\n</ol>\n`;
        //     }
        
        //     // Replace block code first (supporting both backticks and tildes)
        //     text = text.replace(blockCodePattern, (match, lang, p1) => {
        //         const languageClass = lang ? ` class="${lang}"` : '';
        //         return `\n<pre${languageClass}>\n${escapeHtml(p1.trim())}\n</pre>\n`;
        //     });
        
        //     text = text.replace(tildeBlockCodePattern, (match, lang, p1) => {
        //         const languageClass = lang ? ` class="${lang}"` : '';
        //         return `\n<pre${languageClass}>\n${escapeHtml(p1.trim())}\n</pre>\n`;
        //     });
        
        //     // Then replace inline code
        //     text = text.replace(inlineCodePattern, (match, p1) => {
        //         return `<code>${escapeHtml(p1.trim())}</code>`;
        //     });
        
        //     // Replace double newlines with paragraph breaks
        //     text = text.replace(/\n{2,}/g, '\n\n');
        
        //     // Add <p> tags around paragraphs
        //     text = text.split('\n\n').map(paragraph => {
        //         return `<p>${paragraph.trim()}</p>`;
        //     }).join('\n');
        
        //     return text.trim(); // Return the final processed text
        // }

        // // Example usage
        // const inputText = `
        // # Heading Level 1
        // Here is some inline code: \`const x = 10;\`

        // ## Heading Level 2
        // And here is a JavaScript code block:

        // \`\`\`javascript
        // function test() {
        //     console.log("Hello, World!");
        // }
        // \`\`\`

        // ### Heading Level 3
        // Here is a bullet list:
        // * Item 1
        //     * Sub-item 1.1
        //     * Sub-item 1.2
        // * Item 2

        // Here is a numbered list:
        // 1. First item
        // 2. Second item
        //    1. Sub-item 2.1
        //    2. Sub-item 2.2

        // Here is a Python code block:

        // ~~~python
        // def greet():
        //     print("Hello, World!")
        // ~~~
        // `;

        // console.log(formatCode(inputText));































    
        // function escapeHtml(html) {
        //     // Escape HTML characters to prevent XSS attacks
        //     const div = document.createElement('div');
        //     div.innerText = html;
        //     return div.innerHTML;
        // }