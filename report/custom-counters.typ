// Custom counters for equations and tables (separate from figure counter)
#let eq-counter = counter("equation")
#let table-counter = counter("table")

// Custom equation function with its own counter
#let eq-numbered(content, caption: none, label-name: none) = [
  #eq-counter.step()
  #context {
    let eq-num = eq-counter.get().first()
    let fig = figure(
      content,
      supplement: [Equation],
      caption: caption,
      numbering: n => str(eq-num)
    )
    // If label-name is provided, attach label to the figure
    if label-name != none {
      [#fig #label-name]
    } else {
      fig
    }
  }
]

// Custom table function with its own counter
#let table-numbered(content, caption: none, label-name: none) = [
  #table-counter.step()
  #context {
    let table-num = table-counter.get().first()
    let fig = figure(
      content,
      supplement: [Table],
      caption: caption,
      numbering: n => str(table-num)
    )
    // If label-name is provided, attach label to the figure
    if label-name != none {
      [#fig #label-name]
    } else {
      fig
    }
  }
]
