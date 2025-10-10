"""Since you know how to use pydantic it automatically does schema validation from the body"""

'''first you make a pydantic base class'''
from pydantic import BaseModel
class RequestModel(BaseModel):
    prompt: str
    metadata: dict

''' this is an example of how to define a base schema'''
'''After we define the endpoint using the pydantic base class schema or we use the body functon in fastapi'''
@app.post('/')
async def endpoint(data: RequestModel):
    "You can then access the receved prompt or data in the form of a dict"
    prompt = data.prompt
    metadata = data.metadata
    results = call_ai(prompt=prompt)
    "then feed the prompt and metadata into the ai when calling the model, the ai would then return the data we need so we return that data from the endpoint"
    return results

''' You could also use the normal body method like u did for query'''
from fastapi import Body
async def endpoint(prompt: str = Body(...), 
                   metadata: dict =  Body(...)):
    "You can then access the receved prompt or data in the form of a dict"
    prompt = prompt
    metadata = metadata
    results = call_ai(prompt=prompt)
    "then feed the prompt and metadata into the ai when calling the model, the ai would then return the data we need so we return that data from the endpoint"
    return results