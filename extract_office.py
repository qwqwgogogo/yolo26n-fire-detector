import zipfile, re, sys, os
import xml.etree.ElementTree as ET

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
S = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def docx_text(path):
    out = []
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8', 'ignore')
    # tables: capture cell text grouped
    root = ET.fromstring(xml)
    body = root.find(W+'body')
    def para_text(p):
        return ''.join(t.text or '' for t in p.iter(W+'t'))
    for el in body:
        tag = el.tag
        if tag == W+'p':
            txt = para_text(el)
            if txt.strip():
                out.append(txt)
        elif tag == W+'tbl':
            out.append('  [TABLE]')
            for row in el.iter(W+'tr'):
                cells = []
                for tc in row.iter(W+'tc'):
                    ct = ''.join(t.text or '' for t in tc.iter(W+'t'))
                    cells.append(ct.strip())
                out.append('   | ' + ' | '.join(cells))
            out.append('  [/TABLE]')
    return '\n'.join(out)

def xlsx_text(path):
    out = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        # shared strings
        shared = []
        if 'xl/sharedStrings.xml' in names:
            sx = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in sx.iter(S+'si'):
                txt = ''.join(t.text or '' for t in si.iter(S+'t'))
                shared.append(txt)
        # sheet files
        sheets = sorted([n for n in names if re.match(r'xl/worksheets/sheet\d+\.xml$', n)])
        for sh in sheets:
            out.append(f'=== {sh} ===')
            rx = ET.fromstring(z.read(sh))
            for row in rx.iter(S+'row'):
                cells = []
                for c in row.iter(S+'c'):
                    t = c.get('t')
                    v = c.find(S+'v')
                    isv = c.find(S+'is')
                    if t == 's' and v is not None:
                        val = shared[int(v.text)] if v.text else ''
                    elif isv is not None:
                        val = ''.join(tt.text or '' for tt in isv.iter(S+'t'))
                    elif v is not None:
                        val = v.text or ''
                    else:
                        val = ''
                    cells.append(val)
                if any(cells):
                    out.append(' | '.join(cells))
    return '\n'.join(out)

for p in sys.argv[1:]:
    print('\n\n########## FILE:', os.path.basename(p), '##########')
    try:
        if p.lower().endswith('.docx'):
            print(docx_text(p))
        elif p.lower().endswith('.xlsx'):
            print(xlsx_text(p))
    except Exception as e:
        print('ERR', e)
