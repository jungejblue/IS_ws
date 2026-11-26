"""Image stitching assignment: single-file implementation.

Q1: SIFT/RANSAC planar; Q2: SIFT, cylindrical APAP and DIS seam refinement.
Q3/E1/E2: AKAZE/MAGSAC++, joint camera optimization, APAP-style warp.
All images are loaded only from img_paths; functions return BGR uint8 arrays.
"""
import cv2
import numpy as np


def _normalize(p):
    center = p.mean(axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(p-center, axis=1))
    T = np.array([[scale, 0, -scale*center[0]], [0, scale, -scale*center[1]], [0, 0, 1.]])
    return (p-center)*scale, T

def _moving_dlt(query, source, target, sigma=90., floor=.01):
    x, T = _normalize(source)
    y, U = _normalize(target)
    X = np.c_[x, np.ones(len(x))]
    z = np.zeros_like(X)
    a = np.concatenate((X, z, -y[:, 0:1]*X), axis=1)
    b = np.concatenate((z, X, -y[:, 1:2]*X), axis=1)
    outer = (np.einsum('ni,nj->nij', a, a)+np.einsum('ni,nj->nij', b, b)).reshape(-1, 81)
    result = []
    for start in range(0, len(query), 128):
        q = query[start:start+128]
        weights = np.maximum(np.exp(-np.sum((q[:, None]-source)**2, axis=2)/sigma**2), floor)
        gram = (weights**2 @ outer).reshape(-1, 9, 9)
        _, vec = np.linalg.eigh(gram)
        H = np.linalg.inv(U)[None] @ vec[:, :, 0].reshape(-1, 3, 3) @ T[None]
        v = np.einsum('nij,nj->ni', H, np.c_[q, np.ones(len(q))])
        result.append(v[:, :2]/v[:, 2:3])
    return np.concatenate(result)





def matcher(images):
    det=cv2.AKAZE_create();features=[det.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None) for im in images]
    if any(d is None for _,d in features):raise ValueError('Image has no descriptors')
    bf=cv2.BFMatcher(cv2.NORM_HAMMING);edges=[];info={}
    for i in range(8):
      for j in range(i+1,8):
        if max(abs(i//4-j//4),abs(i%4-j%4))!=1:continue
        ka,da=features[i];kb,db=features[j];ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2)
        rev={m.queryIdx:m.trainIdx for z in ba if len(z)==2 for m,n in [z] if m.distance<.75*n.distance}
        good=[m for z in ab if len(z)==2 for m,n in [z] if m.distance<.75*n.distance and rev.get(m.trainIdx)==m.queryIdx]
        inf={'candidates':len(good)};info[f'{i+1}-{j+1}']=inf
        if len(good)<12:inf['reason']='fewer than 12 candidates';continue
        a=np.float64([ka[m.queryIdx].pt for m in good]);b=np.float64([kb[m.trainIdx].pt for m in good])
        cv2.setRNGSeed(42);H,mask=cv2.findHomography(a,b,cv2.USAC_MAGSAC,4.,maxIters=5000,confidence=.995)
        if H is None or mask is None or int(mask.sum())<12:inf['reason']='fewer than 12 MAGSAC inliers';continue
        yes=mask.ravel()>0;a=a[yes];b=b[yes]
        projected=cv2.perspectiveTransform(a.reshape(-1,1,2),H).reshape(-1,2)
        err=np.linalg.norm(projected-b,axis=1);delta=np.median(b-a,axis=0)
        area=cv2.contourArea(cv2.convexHull(a.astype(np.float32)))/(images[i].shape[0]*images[i].shape[1])*100
        inf.update(inliers=len(a),fit_median_px=float(np.median(err)),delta=delta.tolist(),hull_percent=area)
        edges.append((i,j,a,b))
    return features,edges,info

def get_rotations(edges,images):
    h,w=images[0].shape[:2];f0=w*1.1;K=np.array([[f0,0,(w-1)/2],[0,f0,(h-1)/2],[0,0,1.]])
    rel=[]
    for i,j,a,b in edges:
        H,mask=cv2.findHomography(a,b,cv2.USAC_MAGSAC,4.0)
        if H is None:continue
        X=np.linalg.inv(K)@H@K
        if np.linalg.det(X)<0:X=-X
        u,_,v=np.linalg.svd(X); r=u@v
        if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@v
        rel.append((i,j,r,len(a)))
    rotations={1:np.eye(3)}
    while len(rotations)<len(images):
        choices=sorted((x for x in rel if (x[0] in rotations) != (x[1] in rotations)),key=lambda x:x[3],reverse=True)
        if not choices:raise ValueError('camera graph disconnected')
        i,j,r,_=choices[0]
        if i in rotations:rotations[j]=rotations[i]@r.T
        else:rotations[i]=rotations[j]@r
    indices=[i for i in range(8) if i!=1];x0=np.concatenate([np.array([np.log(f0)])]+[cv2.Rodrigues(rotations[i])[0].ravel() for i in indices])
    edges_small=[]
    for i,j,a,b in edges:
        ids=np.linspace(0,len(a)-1,min(len(a),200),dtype=int);edges_small.append((i,j,a[ids],b[ids]))
    def compute(x):
        f=np.exp(x[0]); mats={1:np.eye(3)}
        for k,i in enumerate(indices):mats[i]=cv2.Rodrigues(x[1+3*k:4+3*k])[0]
        residual=[]
        for i,j,a,b in edges_small:
            def rays(p):
                z=np.c_[(p[:,0]-(w-1)/2)/f,(p[:,1]-(h-1)/2)/f,np.ones(len(p))]
                return z/np.linalg.norm(z,axis=1)[:,None]
            diff=(rays(a)@mats[i].T-rays(b)@mats[j].T)*f
            residual.append((diff/np.sqrt(len(a))).ravel())
        residual.append(np.array([(x[0]-np.log(f0))*8]))
        return np.concatenate(residual)
    lower=np.r_[np.log(w*.45),np.full(21,-np.inf)];upper=np.r_[np.log(w*3.),np.full(21,np.inf)]
    opt=_least_squares(compute,x0,lower,upper)
    f=float(np.exp(opt.x[0]));mats=[np.eye(3) for _ in images]
    for k,i in enumerate(indices):mats[i]=cv2.Rodrigues(opt.x[1+3*k:4+3*k])[0]
    corrected=cv2.detail.waveCorrect([m.astype(np.float32) for m in mats[:4]],cv2.detail.WAVE_CORRECT_HORIZ)
    common_rotation=corrected[0]@mats[0].T
    mats=[common_rotation@m for m in mats]
    # Center the view away from the spherical longitude wrap.
    forward=np.sum([m[:3,2] for m in mats[:4]],axis=0);angle=np.arctan2(forward[0],forward[2]);R=cv2.Rodrigues(np.array([0.,-angle,0.]))[0]
    mats=[(R@m).astype(np.float32) for m in mats]
    return f,mats,{'initial_f':f0,'f':f,'iterations':int(opt.nfev),'success':bool(opt.success),'cost':float(opt.cost),'camera_directions':[m[:,2].tolist() for m in mats]}

def project(images,edges,f,mats,use_apap,projection='spherical'):
    warper=cv2.PyRotationWarper(projection,float(f));layers=[];projected=[]
    for idx,im in enumerate(images):
        h,w=im.shape[:2];K=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]],np.float32)
        corner,warped=warper.warp(im,K,mats[idx],cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)
        _,mask=warper.warp(np.full((h,w),255,np.uint8),K,mats[idx],cv2.INTER_NEAREST,cv2.BORDER_CONSTANT)
        layers.append({'image':warped,'mask':mask,'corner':np.array(corner),'K':K})
    target=[[] for _ in images];source=[[] for _ in images]
    for i,j,a,b in edges:
        def points(k,p):
            return np.array([warper.warpPoint(tuple(map(float,pt)),layers[k]['K'],mats[k]) for pt in p])
        u,v=points(i,a),points(j,b)
        # Reject longitude-wrap/outlying pair disagreements to protect local warp.
        distance=np.linalg.norm(u-v,axis=1);keep=distance<min(250.,np.pi*f/3)
        if keep.sum()<8:continue
        u,v=u[keep],v[keep];mid=(u+v)/2
        source[i].append(u-layers[i]['corner']);target[i].append(mid)
        source[j].append(v-layers[j]['corner']);target[j].append(mid)
        projected.append({'pair':f'{i+1}-{j+1}','count':int(keep.sum()),'initial_median_px':float(np.median(distance[keep]))})
    for i,layer in enumerate(layers):
        if not use_apap or not source[i]:continue
        image,mask,origin=layer['image'],layer['mask'],layer['corner'];h,w=mask.shape
        src=np.concatenate(source[i]);dst=np.concatenate(target[i]);ax,ay=np.meshgrid(np.linspace(0,w-1,6),np.linspace(0,h-1,5));anchors=np.c_[ax.ravel(),ay.ravel()]
        src=np.vstack([src,anchors]);dst=np.vstack([dst,anchors+origin])
        gx=max(3,int(np.ceil(w/32))+1);gy=max(3,int(np.ceil(h/32))+1);x,y=np.meshgrid(np.linspace(0,w-1,gx),np.linspace(0,h-1,gy));q=np.c_[x.ravel(),y.ravel()]
        try:
            mapped=_moving_dlt(q+origin,dst,src,sigma=120.,floor=.01);delta=mapped-q;delta[~np.isfinite(delta).all(1)]=0
            n=np.linalg.norm(delta,axis=1);delta*=np.minimum(1.,100/np.maximum(n,1e-9))[:,None]
            yy,xx=np.indices((h,w),np.float32);field=delta.reshape(gy,gx,2).astype(np.float32);layer['field']=field;dd=cv2.remap(field,xx*(gx-1)/(w-1),yy*(gy-1)/(h-1),cv2.INTER_LINEAR)
            mx=xx+dd[:,:,0];my=yy+dd[:,:,1]
            layer['image']=cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            layer['mask']=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        except (ValueError,np.linalg.LinAlgError):pass
    return layers,projected

def render_sphere(layers):
    corners=np.array([l['corner'] for l in layers]);ends=np.array([l['corner']+np.array([l['mask'].shape[1],l['mask'].shape[0]]) for l in layers]);lo=corners.min(0);hi=ends.max(0);W,H=(hi-lo).tolist()
    if W*H>40000000:raise ValueError('unreasonable sphere canvas')
    base=np.zeros((H,W,3),np.uint8);best=np.full((H,W),-1.,np.float32);labels=np.zeros((H,W),np.uint8)
    for i,l in enumerate(layers):
        x,y=(l['corner']-lo).tolist();im=l['image'];mask=l['mask'];h,w=mask.shape
        padded=cv2.copyMakeBorder((mask>0).astype(np.uint8),1,1,1,1,cv2.BORDER_CONSTANT);weight=cv2.distanceTransform(padded,cv2.DIST_L2,3)[1:-1,1:-1]
        bb=best[y:y+h,x:x+w];sel=(mask>0)&(weight>bb);bb[sel]=weight[sel];base[y:y+h,x:x+w][sel]=im[sel];labels[y:y+h,x:x+w][sel]=i+1
    blender=cv2.detail_MultiBandBlender();blender.setNumBands(5);blender.prepare((0,0,W,H))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41))
    for i,l in enumerate(layers):
        x,y=(l['corner']-lo).tolist();im=l['image'];mask=l['mask'];h,w=mask.shape
        ownership=(labels[y:y+h,x:x+w]==i+1).astype(np.uint8)*255
        seam_mask=cv2.dilate(ownership,kernel)&mask
        blender.feed(im.astype(np.int16),seam_mask,(x,y))
    blended,blendmask=blender.blend(None,None);blended=np.clip(blended,0,255).astype(np.uint8);blended[blendmask==0]=0
    sy,sx=np.nonzero(labels);x0,x1=sx.min(),sx.max()+1;y0,y1=sy.min(),sy.max()+1
    return base[y0:y1,x0:x1],blended[y0:y1,x0:x1],labels[y0:y1,x0:x1],lo+np.array([x0,y0])


def _least_squares(fun, x0, lower, upper):
    """Damped finite-difference least squares with soft-L1 weights."""
    from types import SimpleNamespace
    x=x0.copy(); damping=1e-2; count=0
    def cost(r): return float(np.sum(np.sqrt(1+r*r)-1))
    r=fun(x); c=cost(r)
    for iteration in range(35):
        jac=np.empty((len(r),len(x)))
        for j in range(len(x)):
            step=1e-5*max(1.,abs(x[j])); xp=x.copy(); xp[j]+=step
            jac[:,j]=(fun(xp)-r)/step
        weight=(1+r*r)**(-.25); jw=jac*weight[:,None]; rw=r*weight
        a=jw.T@jw; g=jw.T@rw
        accepted=False
        for _ in range(8):
            try: delta=np.linalg.solve(a+damping*np.diag(np.maximum(np.diag(a),1e-8)),-g)
            except np.linalg.LinAlgError: break
            trial=np.clip(x+delta,lower,upper); rr=fun(trial); cc=cost(rr); count+=1
            if np.isfinite(cc) and cc<c:
                x,r,c=trial,rr,cc; damping=max(damping/3,1e-8); accepted=True; break
            damping*=10
        if not accepted or np.linalg.norm(delta)<1e-6: break
    return SimpleNamespace(x=x,nfev=count,success=True,cost=c)


def _read_images(paths, count):
    if len(paths)!=count: raise ValueError(f'Expected {count} images; got {len(paths)}')
    images=[]
    for path in paths:
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f'Cannot read image: {path}')
        scale=min(1.,1400/max(image.shape[:2]))
        if scale<1: image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        images.append(image)
    return images


def q1(img_paths):
    """Planar SIFT and RANSAC registration of two images."""
    a,b=_read_images(img_paths,2)
    detector=cv2.SIFT_create(nfeatures=6000)
    ka,da=detector.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None)
    kb,db=detector.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None or min(len(da),len(db))<2: raise ValueError('Insufficient SIFT features')
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
    if len(good)<10: raise ValueError('Insufficient SIFT matches')
    src=np.float32([ka[m.queryIdx].pt for m in good]); dst=np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(42); H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=5000,confidence=.995)
    if H is None or mask is None or mask.sum()<10: raise ValueError('Homography estimation failed')
    h,w=a.shape[:2]; h2,w2=b.shape[:2]
    corners=np.float32([[0,0],[w2-1,0],[w2-1,h2-1],[0,h2-1]])
    pts=cv2.perspectiveTransform(corners.reshape(-1,1,2),np.linalg.inv(H)).reshape(-1,2)
    bounds=np.vstack([[[0,0],[w-1,h-1]],pts]); lo=np.floor(bounds.min(0)).astype(int); hi=np.ceil(bounds.max(0)).astype(int)
    W,Hh=(hi-lo+1).tolist()
    if min(W,Hh)<=0 or W*Hh>15000000: raise ValueError('Invalid panorama bounds')
    shift=np.array([[1,0,-lo[0]],[0,1,-lo[1]],[0,0,1.]],float)
    im_a=cv2.warpPerspective(a,shift,(W,Hh)); im_b=cv2.warpPerspective(b,shift@np.linalg.inv(H),(W,Hh))
    ma=cv2.warpPerspective(np.full((h,w),255,np.uint8),shift,(W,Hh),flags=cv2.INTER_NEAREST)>0
    mb=cv2.warpPerspective(np.full((h2,w2),255,np.uint8),shift@np.linalg.inv(H),(W,Hh),flags=cv2.INTER_NEAREST)>0
    wa=cv2.distanceTransform(ma.astype(np.uint8),cv2.DIST_L2,3); wb=cv2.distanceTransform(mb.astype(np.uint8),cv2.DIST_L2,3)
    res=np.zeros_like(im_a); res[ma]=im_a[ma]; take=mb&(~ma|(wb>wa));res[take]=im_b[take]
    return res


def _multi(img_paths, projection, blend):
    images=_read_images(img_paths,8)
    if any(im.shape!=images[0].shape for im in images): raise ValueError('Images must have the same size')
    _,edges,_=matcher(images)
    focal,rotations,_=get_rotations(edges,images)
    layers,_=project(images,edges,focal,rotations,True,projection=projection)
    base,blended,_,_=render_sphere(layers)
    return blended if blend else base


def q3(img_paths): return _multi(img_paths,'cylindrical',False)
def e1(img_paths): return _multi(img_paths,'spherical',False)
def e2(img_paths): return _multi(img_paths,'spherical',True)


# Q2: SIFT + camera model + APAP and LPAM-inspired DIS seam refinement.
import sys
import time
from collections import deque

def transform(p, H):
    hp = np.column_stack((p, np.ones(len(p)))) @ H.T
    return hp[:, :2] / hp[:, 2:3]

def load_images(paths, max_side=1400):
    images = []
    for p in paths:
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError('Cannot read ' + str(p))
        scale = min(1., max_side / max(im.shape[:2]))
        if scale < 1:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(im)
    if len(images) != 3 or any(im.shape != images[1].shape for im in images):
        raise ValueError('This Q2 experiment expects three equally sized images')
    return images

def get_matches(images):
    detector = cv2.SIFT_create(nfeatures=4000)
    kps, desc = [], []
    for im in images:
        gray = cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        k, d = detector.detectAndCompute(gray, None)
        if d is None:
            raise ValueError('No SIFT descriptors')
        kps.append(k)
        desc.append(d)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = {}
    for i in (0, 2):
        ab = matcher.knnMatch(desc[i], desc[1], k=2)
        ba = matcher.knnMatch(desc[1], desc[i], k=2)
        reverse = {x[0].queryIdx: x[0].trainIdx for x in ba
                   if len(x) == 2 and x[0].distance < .75*x[1].distance}
        good = [x[0] for x in ab if len(x) == 2 and x[0].distance < .75*x[1].distance
                and reverse.get(x[0].trainIdx) == x[0].queryIdx]
        a = np.array([kps[i][m.queryIdx].pt for m in good], np.float64)
        b = np.array([kps[1][m.trainIdx].pt for m in good], np.float64)
        # Frozen split before robust estimation; no validation correspondences fit models.
        rng = np.random.default_rng(20261115+i)
        ids = rng.permutation(len(a))
        train, test = ids[:int(.75*len(ids))], ids[int(.75*len(ids)):]
        cv2.setRNGSeed(42)
        H, ok = cv2.findHomography(a[train], b[train], cv2.RANSAC, 8.,
                                   maxIters=5000, confidence=.999)
        if H is None or int(ok.sum()) < 30:
            raise ValueError('Insufficient robust matches')
        train = train[ok.ravel() > 0]
        # Loose, common validation filter. This is NOT independent ground truth.
        test = test[np.linalg.norm(transform(a[test], H)-b[test], axis=1) < 12.]
        pairs[i] = dict(a=a[train], b=b[train], va=a[test], vb=b[test], H=H,
                        mutual_ratio_matches=len(good), train_count=len(train),
                        validation_count=len(test))
    return pairs, [len(k) for k in kps]

def fit_rotation(a, b, f, center):
    ar = np.column_stack(((a-center)/f, np.ones(len(a))))
    br = np.column_stack(((b-center)/f, np.ones(len(b))))
    ar /= np.linalg.norm(ar, axis=1)[:, None]
    br /= np.linalg.norm(br, axis=1)[:, None]
    weights = np.ones(len(a))
    for _ in range(15):
        U, _, V = np.linalg.svd((br*weights[:, None]).T @ ar)
        R = U @ np.diag((1., 1., np.linalg.det(U@V))) @ V
        e = np.linalg.norm(ar@R.T-br, axis=1)
        weights = 1./np.maximum(1., e/max(.002, np.median(e)*1.5))
    return R

def camera_solution(pairs, shape):
    h, w = shape[:2]
    center = np.array(((w-1)/2, (h-1)/2))
    def residual(v):
        f = np.exp(v[0])
        allr = []
        for n, i in enumerate((0, 2)):
            R = cv2.Rodrigues(v[1+3*n:4+3*n])[0]
            a, b = pairs[i]['a'], pairs[i]['b']
            ray = np.column_stack(((a-center)/f, np.ones(len(a)))) @ R.T
            allr.append((f*ray[:, :2]/ray[:, 2:3]+center-b).ravel())
        return np.concatenate(allr)
    candidates = []
    for ratio in (.55, .65, .75, .9, 1.1, 1.4):
        f = ratio*w
        v = np.concatenate(([np.log(f)], *[cv2.Rodrigues(
            fit_rotation(pairs[i]['a'], pairs[i]['b'], f, center))[0].ravel()
            for i in (0, 2)]))
        r = residual(v).reshape(-1, 2)
        candidates.append((float(np.median(np.linalg.norm(r, axis=1))), v))
    v = min(candidates, key=lambda x:x[0])[1]
    damping = .01
    def robust_cost(r):
        e = np.linalg.norm(r.reshape(-1, 2), axis=1)
        return float(np.sum(np.where(e <= 4, .5*e*e, 4*(e-2))))
    for _ in range(45):
        r = residual(v)
        e = np.linalg.norm(r.reshape(-1, 2), axis=1)
        weights = np.repeat(np.minimum(1., 4./np.maximum(e, 1e-9)), 2)
        J = np.column_stack([(residual(v+np.eye(7)[j]*1e-5)-r)/1e-5 for j in range(7)])
        A = J.T@(weights[:, None]*J)
        step = np.linalg.solve(A+damping*np.diag(np.maximum(np.diag(A), 1.)),
                               -J.T@(weights*r))
        proposal = v+step
        proposal[0] = np.clip(proposal[0], np.log(.45*w), np.log(2*w))
        if robust_cost(residual(proposal)) < robust_cost(r):
            v = proposal
            damping = max(damping/3, 1e-7)
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            damping = min(damping*5, 1e8)
    f = float(np.exp(v[0]))
    rotations = [cv2.Rodrigues(v[1:4])[0], np.eye(3), cv2.Rodrigues(v[4:7])[0]]
    return f, rotations, dict(focal_px=f, focal_ratio=f/w,
        train_median_camera_error_px=float(np.median(np.linalg.norm(
            residual(v).reshape(-1, 2), axis=1))),
        assumption='shared focal; pure rotation approximation; not calibrated ground truth')

def to_cylinder(p, R, f, shape):
    h, w = shape[:2]
    center = np.array(((w-1)/2, (h-1)/2))
    rays = np.column_stack(((p-center)/f, np.ones(len(p)))) @ R.T
    return f*np.column_stack((np.arctan2(rays[:, 0], rays[:, 2]),
                             rays[:, 1]/np.hypot(rays[:, 0], rays[:, 2])))

def cylinder_setup(images, pairs, f, rotations):
    perimeters = []
    h, w = images[1].shape[:2]
    x, y = np.linspace(0, w-1, 400), np.linspace(0, h-1, 400)
    perimeter = np.concatenate((np.column_stack((x, x*0)),
                                np.column_stack((x, x*0+h-1)),
                                np.column_stack((y*0, y)),
                                np.column_stack((y*0+w-1, y))))
    for R in rotations:
        perimeters.append(to_cylinder(perimeter, R, f, images[1].shape))
    perimeters = np.concatenate(perimeters)
    origin = np.floor(perimeters.min(axis=0))-60
    upper = np.ceil(perimeters.max(axis=0))+60
    size = (upper-origin+1).astype(int)
    if np.prod(size) > 15000000:
        raise ValueError('Unexpected panorama bounds')
    yy, xx = np.indices((size[1], size[0]), dtype=np.float32)
    identity = np.dstack((xx, yy))
    cpairs = {}
    for i, p in pairs.items():
        cpairs[i] = {key: to_cylinder(p[key], rotations[i if key in ('a','va') else 1],
                        f, images[1].shape)-origin for key in ('a','b','va','vb')}
    return origin, identity, cpairs

def render_raw(image, R, f, origin, inverse):
    # Compose deformation with cylindrical inverse; sample the original once.
    theta = (inverse[:, :, 0]+origin[0])/f
    height = (inverse[:, :, 1]+origin[1])/f
    ray = np.stack((np.sin(theta), height, np.cos(theta)), axis=0)
    camera = (R.T @ ray.reshape(3, -1)).reshape(ray.shape)
    z = camera[2]
    h, w = image.shape[:2]
    mx = (f*camera[0]/np.maximum(z, 1e-8)+(w-1)/2).astype(np.float32)
    my = (f*camera[1]/np.maximum(z, 1e-8)+(h-1)/2).astype(np.float32)
    mask = ((z>0)&(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)).astype(np.uint8)*255
    im = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    im[mask==0] = 0
    return im, mask

def map_homography(identity, H):
    return transform(identity.reshape(-1, 2), H).reshape(identity.shape).astype(np.float32)

def sample_map(field, points):
    return cv2.remap(field, points[:, 0].astype(np.float32).reshape(1,-1),
                     points[:, 1].astype(np.float32).reshape(1,-1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1,2)

def normalized(p):
    center = p.mean(axis=0)
    scale = np.sqrt(2)/np.mean(np.linalg.norm(p-center, axis=1))
    T = np.array([[scale,0,-scale*center[0]],[0,scale,-scale*center[1]],[0,0,1.]])
    return (p-center)*scale, T

def moving_dlt(query, source, target, sigma=90., floor=.01):
    # Source is output/reference here: estimate the inverse warp directly.
    x, T = normalized(source)
    y, U = normalized(target)
    X = np.column_stack((x, np.ones(len(x))))
    zero = np.zeros_like(X)
    a = np.concatenate((X, zero, -y[:, 0:1]*X), axis=1)
    b = np.concatenate((zero, X, -y[:, 1:2]*X), axis=1)
    outer = np.einsum('ni,nj->nij', a, a)+np.einsum('ni,nj->nij', b, b)
    result = []
    for start in range(0, len(query), 128):
        q = query[start:start+128]
        weights = np.maximum(np.exp(-np.sum((q[:,None]-source)**2,axis=2)/sigma**2), floor)
        gram = (weights**2 @ outer.reshape(len(x),81)).reshape(-1,9,9)
        _, eigenvectors = np.linalg.eigh(gram)
        H = np.linalg.inv(U)[None] @ eigenvectors[:,:,0].reshape(-1,3,3) @ T[None]
        v = np.einsum('nij,nj->ni', H, np.column_stack((q,np.ones(len(q)))))
        result.append(v[:,:2]/v[:,2:3])
    return np.concatenate(result)

def apap_field(identity, pair, initial):
    h,w = identity.shape[:2]
    gx,gy = max(3,int(np.ceil(w/28))+1),max(3,int(np.ceil(h/28))+1)
    xx,yy = np.meshgrid(np.linspace(0,w-1,gx),np.linspace(0,h-1,gy))
    q = np.column_stack((xx.ravel(),yy.ravel()))
    mapped = moving_dlt(q,pair['b'],pair['a'])
    base = sample_map(initial,q)
    delta = mapped-base
    invalid = ~np.isfinite(delta).all(axis=1)
    delta[invalid] = 0
    norm = np.linalg.norm(delta,axis=1)
    delta *= np.minimum(1.,40/np.maximum(norm,1e-9))[:,None]
    delta = delta.reshape(gy,gx,2).astype(np.float32)
    # Explicit lattice interpolation with nodes at the coordinates used for fitting.
    mx = identity[:,:,0]*(gx-1)/(w-1)
    my = identity[:,:,1]*(gy-1)/(h-1)
    dense = cv2.remap(delta,mx,my,cv2.INTER_LINEAR)
    return initial+dense,dict(grid=[gx,gy],sigma_px=90.,weight_floor=.01,
                              clipped_nodes=int((norm>40).sum()),invalid_nodes=int(invalid.sum()))

def find_border(mask):
    mask=np.asarray(mask,bool)
    right=np.zeros_like(mask);right[:,:-1]=mask[:,1:]
    left=np.zeros_like(mask);left[:,1:]=mask[:,:-1]
    down=np.zeros_like(mask);down[:-1,:]=mask[1:,:]
    up=np.zeros_like(mask);up[1:,:]=mask[:-1,:]
    return (mask&~right)|(mask&~left)|(mask&~down)|(mask&~up)

def mincut(num_nodes, edge_u, edge_v, weights, seed_a, seed_b):
    """Dinic s-t min-cut, symmetric pair costs and hard seeds; float64."""
    N=num_nodes+2; source=num_nodes; sink=num_nodes+1
    head=[[] for _ in range(N)];to=[];cap=[]
    def add(u,v,forward,backward=0.):
        k=len(to);to.extend((v,u));cap.extend((float(forward),float(backward)))
        head[u].append(k);head[v].append(k+1)
    for u,v,w in zip(edge_u,edge_v,weights):add(int(u),int(v),w,w)
    hard=max(1e4,float(np.sum(weights))*2+1)
    for i in seed_a:add(source,int(i),hard)
    for i in seed_b:add(int(i),sink,hard)
    sys.setrecursionlimit(max(10000,min(N+100,200000)))
    flow=0.;eps=1e-12
    while True:
        level=[-1]*N;level[source]=0;q=deque([source])
        while q:
            u=q.popleft()
            for e in head[u]:
                v=to[e]
                if cap[e]>eps and level[v]<0:level[v]=level[u]+1;q.append(v)
        if level[sink]<0:break
        it=[0]*N
        def send(u,pushed):
            if u==sink:return pushed
            while it[u]<len(head[u]):
                e=head[u][it[u]];v=to[e]
                if cap[e]>eps and level[v]==level[u]+1:
                    out=send(v,min(pushed,cap[e]))
                    if out>eps:cap[e]-=out;cap[e^1]+=out;return out
                it[u]+=1
            return 0.
        while True:
            pushed=send(source,float('inf'))
            if pushed<=eps:break
            flow+=pushed
    reachable=np.zeros(N,bool);reachable[source]=True;q=deque([source])
    while q:
        u=q.popleft()
        for e in head[u]:
            v=to[e]
            if cap[e]>eps and not reachable[v]:reachable[v]=True;q.append(v)
    return np.where(reachable[:num_nodes],0,1).astype(np.uint8),flow

def grid_cut(mask,weight,seed_a,seed_b):
    mask=np.asarray(mask,bool);h,w=mask.shape
    ids=np.full((h,w),-1,np.int32);ids[mask]=np.arange(mask.sum())
    hl=mask[:,:-1]&mask[:,1:];vu=mask[:-1,:]&mask[1:,:]
    u=np.concatenate((ids[:,:-1][hl],ids[:-1,:][vu]))
    v=np.concatenate((ids[:,1:][hl],ids[1:,:][vu]))
    c=np.concatenate(((weight[:,:-1][hl]+weight[:,1:][hl])/2+1e-8,
                      (weight[:-1,:][vu]+weight[1:,:][vu])/2+1e-8))
    a=ids[seed_a&mask];b=ids[seed_b&mask]
    labels,cut=mincut(int(mask.sum()),u,v,c,a,b)
    result=np.zeros((h,w),np.uint8);result[mask]=labels
    return result,cut

def image_difference(a,b,mask):
    d=(a.astype(np.float32)-b.astype(np.float32))/255.
    return np.sqrt(np.mean(d*d,axis=2))*mask

def estimate_seam(a,b,ma,mb):
    overlap=ma&mb
    border=find_border(overlap)
    seed_a=find_border(mb)&border
    seed_b=(~seed_a)&border
    label,cut=grid_cut(overlap,image_difference(a,b,overlap),seed_a,seed_b)
    sa=ma.copy();sb=mb.copy()
    sa[overlap&(label==1)]=False
    sb[overlap&(label==0)]=False
    return sa,sb,cut

def ssim_patch(a,b):
    # skimage structural_similarity defaults: uniform win7, covariance N/(N-1),
    # mean over valid interior and mean over channels; 8-bit images -> [0,1].
    a=a.astype(np.float64)/255.;b=b.astype(np.float64)/255.
    if min(a.shape[:2])<7:return 1.-float(np.mean(np.abs(a-b)))
    avg=lambda im:cv2.boxFilter(im,-1,(7,7),normalize=True,borderType=cv2.BORDER_REFLECT)
    ua,ub=avg(a),avg(b);n=49/48
    va=n*(avg(a*a)-ua*ua);vb=n*(avg(b*b)-ub*ub);cab=n*(avg(a*b)-ua*ub)
    s=((2*ua*ub+.01**2)*(2*cab+.03**2))/((ua*ua+ub*ub+.01**2)*(va+vb+.03**2)+1e-16)
    return float(np.mean(s[3:-3,3:-3]))

def seam_quality(a,b,overlap,points):
    err=np.empty(len(points),np.float64);boxes=np.empty((len(points),4),np.int32)
    for k,(y,x) in enumerate(points):
        y1,y2=max(int(y)-10,0),min(int(y)+11,a.shape[0]);x1,x2=max(int(x)-10,0),min(int(x)+11,a.shape[1])
        boxes[k]=y1,y2,x1,x2
        m=overlap[y1:y2,x1:x2,None]
        err[k]=1-ssim_patch(a[y1:y2,x1:x2]*m,b[y1:y2,x1:x2]*m)
    return err,boxes

def safe_otsu(values):
    values=np.asarray(values,np.float64)
    if values.size==0:return 0.
    if np.allclose(values,values[0]):return float(values[0])
    # scikit-image threshold_otsu default: 256 bins over observed min..max.
    hist,edges=np.histogram(values,bins=256)
    centers=(edges[:-1]+edges[1:])/2
    w1=np.cumsum(hist);w2=np.cumsum(hist[::-1])[::-1]
    sum1=np.cumsum(hist*centers);sum2=np.cumsum((hist*centers)[::-1])[::-1]
    mean1=sum1/np.maximum(w1,1);mean2=sum2/np.maximum(w2,1)
    score=w1[:-1]*w2[1:]*(mean1[:-1]-mean2[1:])**2
    return float(centers[int(np.argmax(score))])

def sigmoid_cost(diff):
    # Author's 0.06-bin histogram and alpha selection.
    bins=np.arange(0.,1.+1e-12,.06);centers=np.arange(.03,1.-.03+1e-12,.06)
    counts,_=np.histogram(diff.ravel(),bins=bins);centers=centers[:len(counts)]
    if counts.sum()==0:alpha=1.
    else:
        prob=counts/counts.sum();mean=np.sum(prob*centers);best=-1;idx=0
        for i in range(len(centers)):
            weight=float(prob[:i+1].sum())
            if not 0<weight<1:continue
            cm=float(np.sum(prob[:i+1]*centers[:i+1]))
            score=(cm-mean*weight)**2/(weight*(1-weight))
            if score>best:best=score;idx=i
        alpha=float(centers[idx]+.03)
    return 1/(1+np.exp(np.clip((4/.06)*(alpha-diff),-80,80)))

def patch_graphcut(a,b,old_a,old_b):
    region=np.ones(old_a.shape,bool);outer=find_border(region)
    sa=find_border(old_a)&outer;sb=find_border(old_b)&outer
    label,cut=grid_cut(region,sigmoid_cost(image_difference(a,b,region)),sa,sb)
    return label==0,label==1,cut

def author_flow(a,b,mask):
    # Same grayscale DIS direction, directional sigmoid and cubic sampling as
    # python_version/lpam_py/flow/optical_flow.py (RGB input converted to BGR).
    h,w=mask.shape
    if min(h,w)<12:return a.copy(),None,dict(status='skipped_small_patch')
    ar=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);br=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(br,ar,None)
    direction=np.array([1.,0.],np.float32)
    if np.all(mask[:,-1]):direction=np.array([-1.,0.],np.float32)
    if np.all(mask[0,:]):direction=np.array([0.,1.],np.float32)
    if np.all(mask[-1,:]):direction=np.array([0.,-1.],np.float32)
    corners=np.array([[0,0],[0,h-1],[w-1,0],[w-1,h-1]],np.float32)
    lo,hi=(corners@direction).min(),(corners@direction).max()
    yy,xx=np.indices((h,w),dtype=np.float32)
    t=(direction[0]*xx+direction[1]*yy-lo)/max(hi-lo,1e-9)
    weight=1/(1+np.exp(-8*(t-.5)))
    mx=np.clip(xx+flow[:,:,0]*weight,0,w-1).astype(np.float32)
    my=np.clip(yy+flow[:,:,1]*weight,0,h-1).astype(np.float32)
    result=cv2.remap(a,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return result,(mx,my),dict(status='ok',median_flow=float(np.median(np.linalg.norm(flow,axis=2))),
                       max_flow=float(np.max(np.linalg.norm(flow,axis=2))))

def seam_improve(a,b,ma,mb,sa,sb):
    overlap=ma&mb
    cross=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    points=np.column_stack(np.nonzero((cv2.dilate(sa.astype(np.uint8),cross)>0)&ma&sb))
    displacement=np.zeros((*ma.shape,2),np.float32)
    if len(points)==0:return sa,sb,a,displacement,[],dict(seam_points=0)
    errors,boxes=seam_quality(a,b,overlap,points)
    meta=dict(seam_points=len(points),quality_mean=float(errors.mean()),quality_max=float(errors.max()))
    if errors.max()<=1.5*errors.mean():return sa,sb,a,displacement,[],meta
    threshold=safe_otsu(errors);meta['otsu']=threshold
    artifact=np.zeros(ma.shape,np.uint8)
    for y1,y2,x1,x2 in boxes[errors>=threshold]:artifact[y1:y2,x1:x2]=1
    kernel=np.ones((10,10),np.uint8)
    # Even-sized structuring elements require distinct anchors to match
    # scipy.ndimage.binary_closing's reflection/origin convention.
    artifact=cv2.dilate(artifact,kernel,anchor=(4,4),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    artifact=cv2.erode(artifact,kernel,anchor=(5,5),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    n,labels,stats,_=cv2.connectedComponentsWithStats(artifact,connectivity=4)
    output=a.copy();sa=sa.copy();sb=sb.copy();patches=[]
    for x,y,w,h,area in stats[1:]:
        x,y,w,h=map(int,(x,y,w,h))
        if min(w,h)<2:continue
        A=output[y:y+h,x:x+w];B=b[y:y+h,x:x+w]
        oa=sa[y:y+h,x:x+w];ob=sb[y:y+h,x:x+w]
        new,maps,info=author_flow(A,B,oa)
        if info['status']!='ok':patches.append(dict(rect=[x,y,w,h],**info));continue
        na,nb,cut=patch_graphcut(new,B,oa,ob)
        if cut is None:patches.append(dict(rect=[x,y,w,h],status='skipped_no_seeds'));continue
        old=displacement[y:y+h,x:x+w].copy()
        yy,xx=np.indices((h,w),dtype=np.float32);mx,my=maps
        prev=cv2.remap(old,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        displacement[y:y+h,x:x+w]=prev+np.dstack((mx-xx,my-yy))
        output[y:y+h,x:x+w]=new;sa[y:y+h,x:x+w]=na;sb[y:y+h,x:x+w]=nb
        patches.append(dict(rect=[x,y,w,h],cut=float(cut),**info))
    return sa,sb,output,displacement,patches,meta

def compose_three(warps,masks,sa,sb):
    layers=[sa[0]&masks[0],sb[0]&sb[2]&masks[1],sa[2]&masks[2]]
    weights=[x.astype(np.float32) for x in layers];total=sum(weights)
    missing=(total==0)&np.logical_or.reduce(masks)
    for j in (1,0,2):
        fill=missing&masks[j];weights[j][fill]=1;missing[fill]=False
    total=sum(weights)
    result=sum(im.astype(np.float32)*weight[:,:,None] for im,weight in zip(warps,weights))/np.maximum(total[:,:,None],1e-9)
    return np.clip(result,0,255).astype(np.uint8),(total>0).astype(np.uint8)*255

def run_q2(img_paths, debug_dir=None, scale=0.2):
    """q2 input paths -> BGR image. Optional debug kept outside submission."""
    images = load_images(img_paths)
    pairs, _ = get_matches(images)
    f, R, camera = camera_solution(pairs, images[1].shape)
    origin, identity, cpairs = cylinder_setup(images, pairs, f, R)
    fields = {1: identity}
    for i in (0, 2):
        H = cv2.findHomography(cpairs[i]['b'], cpairs[i]['a'], 0)[0]
        fields[i], _ = apap_field(identity, cpairs[i], map_homography(identity, H))
    full, mfull = zip(*(render_raw(images[i], R[i], f, origin, fields[i]) for i in range(3)))
    h, w = identity.shape[:2]
    small = (max(1, round(w * scale)), max(1, round(h * scale)))
    warps = [cv2.resize(im, small, interpolation=cv2.INTER_AREA) for im in full]
    masks = [cv2.resize(im, small, interpolation=cv2.INTER_NEAREST) > 0 for im in mfull]
    src = {}
    displacements = {}
    base_a = {}
    base_b = {}
    new_a = {}
    new_b = {}
    meta = {}
    for i in (0, 2):
        t = time.perf_counter()
        sa, sb, cut = estimate_seam(warps[i], warps[1], masks[i], masks[1])
        base_a[i] = sa
        base_b[i] = sb
        na, nb, corrected, disp, patches, quality = seam_improve(warps[i], warps[1], masks[i], masks[1], sa, sb)
        new_a[i] = na
        new_b[i] = nb
        src[i] = corrected
        displacements[i] = disp
        meta[str(i + 1)] = dict(cut=cut, quality=quality, patches=patches, seconds=time.perf_counter() - t)
    before, mask = compose_three(warps, masks, base_a, base_b)
    updated = [src[0], warps[1], src[2]]
    after, _ = compose_three(updated, masks, new_a, new_b)
    full_masks = [im > 0 for im in mfull]
    up_a = {i: cv2.resize(base_a[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for i in (0, 2)}
    up_b = {i: cv2.resize(base_b[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for i in (0, 2)}
    up_new_a = {i: cv2.resize(new_a[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for i in (0, 2)}
    up_new_b = {i: cv2.resize(new_b[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for i in (0, 2)}
    modified = list(full)
    yy, xx = np.indices((h, w), dtype=np.float32)
    for i in (0, 2):
        displacement = cv2.resize(displacements[i], (w, h), interpolation=cv2.INTER_LINEAR)
        displacement[:, :, 0] *= w / small[0]
        displacement[:, :, 1] *= h / small[1]
        modified[i] = cv2.remap(full[i], xx + displacement[:, :, 0], yy + displacement[:, :, 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        moved_mask = cv2.remap(mfull[i], xx + displacement[:, :, 0], yy + displacement[:, :, 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        invalid = ~moved_mask & full_masks[i]
        modified[i][invalid] = full[i][invalid]
    before_full, mask_full = compose_three(full, full_masks, up_a, up_b)
    after_full, mask_after = compose_three(modified, full_masks, up_new_a, up_new_b)
    ys, xs = np.nonzero(mask_full | mask_after)
    crop_full = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    ys, xs = np.nonzero(mask)
    crop_small = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return after_full[crop_full]

def q2(img_paths):
    return run_q2(img_paths)
