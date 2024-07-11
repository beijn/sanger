import requests, os 

class GHAPI(object):
  def __init__(self, repo, token=None):
    if token is None: token = os.getenv('GITHUB_TOKEN', None)
    self.repo=repo; self.token = token

  @staticmethod
  def request(repo, endpoint, token=None, paging=100):
    request = f"https://api.github.com/repos/{repo}/{endpoint}"
    headers= {"Authorization": "token " + token} if token else {}

    if not paging: 
      response = requests.get(request, headers=headers)
      return response.request.path_url, response.json()
    
    request = f"{request}?per_page={paging}&page=1"
    responses = []
    while True:
      response = requests.get(request, headers=headers)
      if not response.json(): break
      if not isinstance(response.json(), list):
        raise RuntimeError(f"Request to GH API failed. Please check that the repo, api endpoint, and access token are valid.\nrequest={request}\nresponse={response.json()}")
      next_page = response.links.get('next', None)
      responses += response.json()
      if not next_page: break
      request = next_page['url']

    return request, responses
  
  @staticmethod
  def ghapi(endpoint, paging=0):
    def decorator(func):
      def wrapper(self, *a, **kw):       
        request, response = GHAPI.request(self.repo, endpoint, self.token, paging=paging)
        try: return func(response, *a, **kw)
        except Exception as e:
          raise RuntimeError(f"Request to GH API failed. Please check that the repo, api endpoint, and access token are valid.\nrequest: {request}\nresponse: {response}\nerror={e.__class__.__name__}, {e}")
      return wrapper
    return decorator
  
  @ghapi('releases/latest')
  def get_latest_release(response): 
    return response["tag_name"]  # type: ignore

  @ghapi('releases', paging=100)
  def get_all_releases(response): 
    # sort by release date 
    response = sorted(response, key=lambda x: x['published_at'], reverse=True)
    return [r['tag_name'] for r in response]  # type: ignore
  

def download(url, filename, overwrite=False):
  import tqdm

  if os.path.isdir(filename): filename = os.path.join(filename, url.split('/')[-1])
  if not overwrite and os.path.isfile(filename): return True
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  try: 
    with open(filename, 'wb') as f:   
      with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))

        with tqdm.tqdm(desc='Download', total=total, unit='B', unit_scale=True, unit_divisor=1024) as pb:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pb.update(len(chunk))
  except Exception as e:
    os.remove(filename)
    raise e
  