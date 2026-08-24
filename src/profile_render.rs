use super::*;

// -------------------------------------------------------------------------
// HTML report
// -------------------------------------------------------------------------

impl Profile {
    pub(super) fn render_html(&self) -> String {
        let o = &self.overview;
        let mut body = String::new();

        body.push_str(&format!(
            "<h1>Data profile</h1><div class=cards>\
             <div class=card><b>{}</b><span>rows</span></div>\
             <div class=card><b>{}</b><span>columns</span></div>\
             <div class=card><b>{:.1}%</b><span>missing</span></div>\
             <div class=card><b>{}</b><span>duplicate rows</span></div>\
             <div class=card><b>{}·{}·{}·{}</b><span>num·cat·date·bool</span></div>\
             </div>",
            o.nrows,
            o.ncols,
            if o.total_cells > 0 {
                100.0 * o.missing_cells as f64 / o.total_cells as f64
            } else {
                0.0
            },
            o.duplicate_rows,
            o.n_numeric,
            o.n_categorical,
            o.n_datetime,
            o.n_boolean,
        ));

        if !self.alerts.is_empty() {
            body.push_str("<h2>Alerts</h2><table><tr><th>Column</th><th>Finding</th><th>Suggested step</th></tr>");
            for a in &self.alerts {
                body.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td class=step>{}</td></tr>",
                    esc(a.column.as_deref().unwrap_or("—")),
                    esc(&a.message),
                    esc(a.suggested),
                ));
            }
            body.push_str("</table>");
        }

        body.push_str("<h2>Columns</h2>");
        for col in &self.columns {
            match col {
                ColumnProfile::Numeric(np) => {
                    body.push_str(&format!(
                        "<div class=col><h3>{} <span class=tag>numeric</span></h3>\
                         <div class=stats>mean {:.3} · std {:.3} · min {:.3} · median {:.3} · max {:.3} · \
                         skew {:.2} · kurtosis {:.2} · missing {} · distinct {} · \
                         outliers {} IQR / {} z</div>{}</div>",
                        esc(&np.name),
                        np.mean, np.std, np.min, np.median, np.max,
                        np.skew, np.kurtosis,
                        np.missing, np.distinct, np.outliers, np.outliers_z,
                        histogram_svg(&np.histogram),
                    ));
                }
                ColumnProfile::Categorical(cp) => {
                    let mut bars = String::new();
                    let top_n = cp.top.first().map(|(_, c)| *c).unwrap_or(1).max(1);
                    for (v, c) in &cp.top {
                        bars.push_str(&format!(
                            "<div class=bar><span class=lab>{}</span>\
                             <span class=track><span class=fill style='width:{:.1}%'></span></span>\
                             <span class=n>{}</span></div>",
                            esc(v),
                            100.0 * *c as f64 / top_n as f64,
                            c,
                        ));
                    }
                    body.push_str(&format!(
                        "<div class=col><h3>{} <span class=tag>categorical</span></h3>\
                         <div class=stats>distinct {} · missing {}</div>{}</div>",
                        esc(&cp.name),
                        cp.distinct,
                        cp.missing,
                        bars,
                    ));
                }
            }
        }

        if !self.correlations.high_pairs.is_empty() {
            body.push_str(
                "<h2>Highly correlated pairs</h2><table><tr><th>Pair</th><th>Pearson r</th><th>Spearman ρ</th></tr>",
            );
            let cols = &self.correlations.columns;
            for (a, b, r) in &self.correlations.high_pairs {
                let rho = match (
                    cols.iter().position(|c| c == a),
                    cols.iter().position(|c| c == b),
                ) {
                    (Some(i), Some(j)) => self.correlations.spearman[i][j],
                    _ => f64::NAN,
                };
                body.push_str(&format!(
                    "<tr><td>{} ~ {}</td><td>{:.3}</td><td>{:.3}</td></tr>",
                    esc(a),
                    esc(b),
                    r,
                    rho,
                ));
            }
            body.push_str("</table>");
        }

        if !self.missingness.co_missing.is_empty() {
            body.push_str(
                "<h2>Columns missing together</h2><table><tr><th>Pair</th><th>phi</th></tr>",
            );
            for (a, b, phi) in &self.missingness.co_missing {
                body.push_str(&format!(
                    "<tr><td>{} ~ {}</td><td>{:.3}</td></tr>",
                    esc(a),
                    esc(b),
                    phi,
                ));
            }
            body.push_str("</table>");
        }

        if let Some(t) = &self.target {
            body.push_str(&format!("<h2>Target · {}</h2>", esc(&t.name)));
            match &t.kind {
                TargetKind::Classification { classes } => {
                    body.push_str("<table><tr><th>Class</th><th>Count</th></tr>");
                    for (c, n) in classes {
                        body.push_str(&format!("<tr><td>{}</td><td>{}</td></tr>", esc(c), n));
                    }
                    body.push_str("</table>");
                }
                TargetKind::Regression { correlations } => {
                    body.push_str("<table><tr><th>Feature</th><th>corr with target</th></tr>");
                    for (c, r) in correlations {
                        body.push_str(&format!("<tr><td>{}</td><td>{:.3}</td></tr>", esc(c), r));
                    }
                    body.push_str("</table>");
                }
            }
        }

        format!("<!doctype html><html><head><meta charset=utf-8><title>Data profile</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")
    }
}

const CSS: &str = "\
:root{--ink:#16191c;--soft:#556069;--line:#dce3e8;--oxide:#bc4a1e;--patina:#2c8272;--bg:#f5f8f9}\
body{margin:0;background:#eef1f3;color:var(--ink);font:16px/1.6 system-ui,sans-serif}\
main{max-width:900px;margin:0 auto;padding:32px 24px}\
h1{margin:0 0 16px}h2{margin:28px 0 10px;font-size:1.3rem}h3{margin:0 0 6px;font-size:1rem}\
.cards{display:flex;flex-wrap:wrap;gap:10px}\
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:90px}\
.card b{display:block;font-size:1.3rem;color:var(--oxide)}.card span{font-size:.75rem;color:var(--soft)}\
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:.9rem}\
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}\
th{background:var(--bg);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--soft)}\
.step{font-family:ui-monospace,monospace;color:var(--oxide)}\
.col{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:10px}\
.tag{font-size:.68rem;color:var(--patina);border:1px solid var(--patina);border-radius:5px;padding:1px 6px;vertical-align:middle}\
.stats{font-size:.82rem;color:var(--soft);margin-bottom:8px}\
.hist{display:flex;align-items:flex-end;gap:2px;height:60px}\
.hist .b{flex:1;background:var(--oxide);opacity:.85;border-radius:2px 2px 0 0;min-height:1px}\
.bar{display:flex;align-items:center;gap:8px;font-size:.8rem;margin:2px 0}\
.bar .lab{width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\
.bar .track{flex:1;background:var(--bg);border-radius:4px;overflow:hidden}\
.bar .fill{display:block;height:10px;background:var(--patina)}\
.bar .n{width:48px;text-align:right;color:var(--soft)}";

fn histogram_svg(bins: &[HistBin]) -> String {
    if bins.is_empty() {
        return String::new();
    }
    let max = bins.iter().map(|b| b.count).max().unwrap_or(1).max(1);
    let mut s = String::from("<div class=hist>");
    for b in bins {
        s.push_str(&format!(
            "<div class=b style='height:{:.1}%' title='[{:.2}, {:.2}) — {}'></div>",
            100.0 * b.count as f64 / max as f64,
            b.lo,
            b.hi,
            b.count,
        ));
    }
    s.push_str("</div>");
    s
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
