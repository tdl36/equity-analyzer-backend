import * as React from 'react';
import {selectedLibraryMetadata} from './processing-library-model.mjs';

// A changed selection invalidates readiness immediately; late prior reads cannot
// confirm a different purpose's sources. Only metadata is read, in bounded batches.
export function useSelectedLibraryMetadata({call,path,ticker,collectionId,selected,revision}) {
 const signature=JSON.stringify([...selected].sort()),key=JSON.stringify([path,ticker,collectionId,signature,revision]);
 const [state,setState]=React.useState({key:null,documents:[],missingIds:[],busy:false,error:''});
 React.useEffect(()=>{
  let current=true;const ids=JSON.parse(signature);
  setState({key,documents:[],missingIds:[],busy:!!ids.length,error:''});
  if(ids.length)(async()=>{
   const documents=[],missingIds=[];
   try{
    for(let i=0;i<ids.length;i+=100){
     if(!current)return;
     const batch=ids.slice(i,i+100);
     const response=await call(path+'?selected='+encodeURIComponent(batch.join(',')));
     const result=selectedLibraryMetadata(response,{ticker,collectionId,selected:batch});
     documents.push(...result.documents);missingIds.push(...result.missingIds);
    }
    if(current)setState({key,documents,missingIds,busy:false,error:''});
   }catch(e){if(current)setState({key,documents:[],missingIds:[],busy:false,error:e.message});}
  })();
  return()=>{current=false;};
 },[key]);
 return state.key===key?state:{key,documents:[],missingIds:[],busy:!!selected.length,error:''};
}
