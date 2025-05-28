# Generate the SPEC document by GrapgRAG
## Upload the technical file
The input file should be one or multiple .txt file
```python
Input_file_path = ./ragtest/input
```
## Run Index process
```
graphrag index --root ./ragtest
```

## Query
e.g.,
```
graphrag query \
--root ./ragtest \
--method global \
--query "What are the top themes in this story?"
```
The *method* could be *global*, *local*