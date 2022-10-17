from rich.progress import Column, Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn


text_column = TextColumn(
      "{task.description} "
    , table_column=Column(ratio=1)
    )

bar_column = BarColumn(
      bar_width=80
    , table_column=Column(ratio=1)
    , style = "bar.finished"
    )

progress = Progress(
    #SpinnerColumn('dots10'),
    "[progress.description]{task.description}",
    bar_column,
    "[progress.percentage]{task.percentage:>3.0f}%",
    "{task.completed:>4.0f}/{task.total:<4.0f}",
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    )
