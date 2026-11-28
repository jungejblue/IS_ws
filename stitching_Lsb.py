
# LPAM seam-cutting adaptation: https://github.com/tlliao/LPAM_seam-cutting
# MIT License
# 
# Copyright (c) 2023 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Image stitching assignment: single-file implementation.

Q1: SIFT/RANSAC planar; Q2: SIFT/RANSAC, cylinder, APAP and seam DIS.
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

def _graphcut_masks(layers, canvas_origin, scale=.20):
    """Find color-based seams on the common surface, then restore mask sizes."""
    images, small_masks, corners = [], [], []
    for layer in layers:
        image, mask = layer["image"], layer["mask"]
        h, w = mask.shape
        corner = layer["corner"] - canvas_origin
        start = np.rint(corner * scale).astype(int)
        stop = np.rint((corner + [w, h]) * scale).astype(int)
        size = tuple(np.maximum(stop - start, 1))
        images.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32))
        small = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        small_masks.append(cv2.UMat(small))
        corners.append(tuple(map(int, start)))
    finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    selected = finder.find(images, corners, small_masks)
    if selected is None:
        selected = small_masks
    masks = []
    for seam, layer in zip(selected, layers):
        valid = layer["mask"]
        small = seam.get() if isinstance(seam, cv2.UMat) else seam
        full = cv2.resize(small, valid.shape[::-1], interpolation=cv2.INTER_NEAREST)
        masks.append((full > 0) & (valid > 0))
    return masks


def render_sphere(layers, *, use_graphcut=False, make_blend=True):
    """Compose already registered inputs, with an optional graph-cut seam."""
    corners = np.array([layer["corner"] for layer in layers])
    ends = np.array([layer["corner"] + layer["mask"].shape[::-1] for layer in layers])
    origin = corners.min(axis=0)
    width, height = (ends.max(axis=0) - origin).tolist()
    if min(width, height) < 1 or width * height > 40_000_000:
        raise ValueError("Invalid panorama canvas")

    labels = np.zeros((height, width), dtype=np.uint8)
    best = np.full((height, width), -1., dtype=np.float32)
    if use_graphcut:
        candidates = _graphcut_masks(layers, origin)
        cut_labels = np.zeros_like(labels)
        cut_best = np.full_like(best, -1.)
    for i, layer in enumerate(layers, start=1):
        x, y = (layer["corner"] - origin).tolist()
        valid = layer["mask"] > 0
        h, w = valid.shape
        padded = cv2.copyMakeBorder(valid.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        region = np.s_[y:y + h, x:x + w]
        take = valid & (distance > best[region])
        best[region][take] = distance[take]
        labels[region][take] = i
        if use_graphcut:
            take = candidates[i - 1] & (distance > cut_best[region])
            cut_best[region][take] = distance[take]
            cut_labels[region][take] = i
    if use_graphcut:
        # Resolve holes from upsampling by keeping the original valid owner.
        keep = cut_labels > 0
        labels[keep] = cut_labels[keep]
    y, x = np.nonzero(labels)
    if not len(x):
        raise ValueError("Empty panorama")
    crop = np.s_[y.min():y.max() + 1, x.min():x.max() + 1]

    base = np.zeros((height, width, 3), dtype=np.uint8)
    if make_blend:
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(5)
        blender.prepare((0, 0, width, height))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    for i, layer in enumerate(layers, start=1):
        x, y = (layer["corner"] - origin).tolist()
        image, mask = layer["image"], layer["mask"]
        h, w = mask.shape
        region = np.s_[y:y + h, x:x + w]
        owns = labels[region] == i
        base[region][owns] = image[owns]
        if make_blend and owns.any():
            seam_mask = cv2.dilate(owns.astype(np.uint8) * 255, kernel) & mask
            valid = mask > 0
            if not valid.all():
                # Fill warp padding before building the Laplacian pyramid so
                # black outside the image cannot darken the blended boundary.
                _, nearest = cv2.distanceTransformWithLabels(
                    (~valid).astype(np.uint8), cv2.DIST_L2, 5,
                    labelType=cv2.DIST_LABEL_PIXEL,
                )
                colors = np.empty((int(nearest.max()) + 1, 3), dtype=image.dtype)
                colors[nearest[valid]] = image[valid]
                image = colors[nearest]
            blender.feed(image.astype(np.int16), seam_mask, (x, y))
    blended = None
    if make_blend:
        blended, blendmask = blender.blend(None, None)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended[blendmask == 0] = 0
        blended = blended[crop]
    return base[crop], blended, labels[crop], origin + np.array([crop[1].start, crop[0].start])


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


def _multi(img_paths, projection, blend, *, graphcut=False):
    images=_read_images(img_paths,8)
    if any(im.shape!=images[0].shape for im in images): raise ValueError('Images must have the same size')
    _,edges,_=matcher(images)
    focal,rotations,_=get_rotations(edges,images)
    layers,_=project(images,edges,focal,rotations,True,projection=projection)
    base,blended,_,_=render_sphere(layers,use_graphcut=graphcut,make_blend=blend)
    return blended if blend else base


def q3(img_paths): return _multi(img_paths,'cylindrical',False)
def e1(img_paths): return _multi(img_paths,'spherical',False)
def e2(img_paths): return _multi(img_paths,'spherical',True,graphcut=True)


def _q2_sift_pairs(images):
    detector = cv2.SIFT_create(nfeatures=4000)
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.append(detector.detectAndCompute(gray, None))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = {}
    for i in (0, 2):
        ka, da = features[i]
        kb, db = features[1]
        if da is None or db is None or min(len(da), len(db)) < 2:
            raise ValueError(f"Insufficient SIFT descriptors for Q2 image {i + 1}")
        backward = matcher.knnMatch(db, da, k=2)
        reverse = {m.queryIdx: m.trainIdx for row in backward if len(row) == 2
                   for m, n in [row] if m.distance < .75 * n.distance}
        forward = matcher.knnMatch(da, db, k=2)
        matches = [m for row in forward if len(row) == 2 for m, n in [row]
                   if m.distance < .75 * n.distance and reverse.get(m.trainIdx) == m.queryIdx]
        if len(matches) < 8:
            raise ValueError(f"Insufficient Q2 matches for image {i + 1}")
        a = np.float64([ka[m.queryIdx].pt for m in matches])
        b = np.float64([kb[m.trainIdx].pt for m in matches])
        cv2.setRNGSeed(42)
        _, inliers = cv2.findHomography(a, b, cv2.RANSAC, 4., maxIters=5000, confidence=.995)
        if inliers is None or inliers.sum() < 8:
            raise ValueError(f"Insufficient RANSAC inliers for Q2 image {i + 1}")
        keep = inliers.ravel() != 0
        pairs[i] = a[keep], b[keep]
    return pairs


def _q2_to_cylinder(points, shape, scale):
    h, w = shape[:2]
    center = np.array([(w - 1) / 2., (h - 1) / 2.])
    x, y = (points - center).T
    return np.column_stack((scale * np.arctan(x / scale),
                            y * scale / np.hypot(x, scale))) + center


def _q2_from_cylinder(points, shape, scale):
    h, w = shape[:2]
    cx, cy = (w - 1) / 2., (h - 1) / 2.
    theta = (points[..., 0] - cx) / scale
    return (scale * np.tan(theta) + cx,
            (points[..., 1] - cy) / np.cos(theta) + cy)


def _q2_cylinder_layers(images, pairs, scale):
    controls = {}
    homographies = [None, np.eye(3), None]
    for i, (a, b) in pairs.items():
        a = _q2_to_cylinder(a, images[i].shape, scale)
        b = _q2_to_cylinder(b, images[1].shape, scale)
        controls[i] = a, b
        # Fit the global part of APAP on the RANSAC inliers in cylinder coordinates.
        h, _ = cv2.findHomography(a, b, 0)
        if h is None or not np.isfinite(h).all():
            raise ValueError(f"Q2 cylinder alignment failed for image {i + 1}")
        homographies[i] = h
    corners = []
    for image, h in zip(images, homographies):
        ih, iw = image.shape[:2]
        x = np.linspace(0, iw - 1, 128)
        y = np.linspace(0, ih - 1, 128)
        boundary = np.vstack((np.column_stack((x, x * 0)),
                              np.column_stack((x, x * 0 + ih - 1)),
                              np.column_stack((y * 0, y)),
                              np.column_stack((y * 0 + iw - 1, y))))
        cylinder = _q2_to_cylinder(boundary, image.shape, scale)
        corners.append(cv2.perspectiveTransform(cylinder[None], h)[0])
    corners = np.concatenate(corners)
    if not np.isfinite(corners).all() or np.abs(corners).max() > 30000:
        raise ValueError("Invalid Q2 cylindrical bounds")
    origin = np.floor(corners.min(axis=0)).astype(int) - 45
    end = np.ceil(corners.max(axis=0)).astype(int) + 45
    w, h = end - origin + 1
    if min(w, h) < 2 or max(w, h) >= 32767 or w * h > 20_000_000:
        raise ValueError("Invalid Q2 cylindrical canvas")
    yy, xx = np.indices((h, w), dtype=np.float32)
    canvas = np.dstack((xx + origin[0], yy + origin[1])).astype(np.float32)
    gx, gy = int(np.ceil(w / 28)) + 1, int(np.ceil(h / 28)) + 1
    xg, yg = np.meshgrid(np.linspace(0, w - 1, gx), np.linspace(0, h - 1, gy))
    grid = np.column_stack((xg.ravel(), yg.ravel()))
    fields, warps, masks = [], [], []
    for i, (image, matrix) in enumerate(zip(images, homographies)):
        inverse = np.linalg.inv(matrix)
        field = cv2.perspectiveTransform(canvas, inverse)
        if i != 1:
            source, target = controls[i]
            local = _moving_dlt(grid, target - origin, source)
            base = cv2.perspectiveTransform((grid + origin)[None], inverse)[0]
            delta = local - base
            delta[~np.isfinite(delta).all(axis=1)] = 0.
            norm = np.linalg.norm(delta, axis=1)
            delta *= np.minimum(1., 40. / np.maximum(norm, 1e-9))[:, None]
            field += cv2.remap(
                delta.reshape(gy, gx, 2).astype(np.float32),
                xx * (gx - 1) / (w - 1), yy * (gy - 1) / (h - 1), cv2.INTER_LINEAR,
            )
        mx, my = _q2_from_cylinder(field, image.shape, scale)
        ih, iw = image.shape[:2]
        angle = np.abs((field[..., 0] - (iw - 1) / 2.) / scale)
        mask = (angle < np.pi / 2) & (mx >= 0) & (my >= 0) & (mx < iw - 1) & (my < ih - 1)
        mx = np.where(mask, mx, -1).astype(np.float32)
        my = np.where(mask, my, -1).astype(np.float32)
        fields.append(field)
        warps.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR))
        masks.append(mask)
    return fields, warps, masks, origin


def _q2_owners(masks):
    """Assign each output pixel to the valid image farthest from its border."""
    owners = np.full(masks[0].shape, -1, dtype=np.int8)
    best = np.zeros(masks[0].shape, dtype=np.float32)
    for i in (1, 0, 2):
        padded = cv2.copyMakeBorder(masks[i].astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        take = distance > best
        owners[take] = i
        best[take] = distance[take]
    return owners


def _q2_dis(warps, masks, owners, radius=40):
    """Align only bands where the reference and a side image meet."""
    result = list(warps)
    offsets = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    center = owners == 1
    touches_center = cv2.dilate(center.astype(np.uint8), kernel) != 0
    for i in (0, 2):
        offset = np.zeros((*owners.shape, 2), dtype=np.float32)
        offsets[i] = offset
        overlap = masks[i] & masks[1]
        seam = (owners == i) & touches_center & overlap
        if not seam.any():
            continue
        y, x = np.nonzero(seam)
        margin = radius + 16
        y0, y1 = max(0, y.min() - margin), min(owners.shape[0], y.max() + margin + 1)
        x0, x1 = max(0, x.min() - margin), min(owners.shape[1], x.max() + margin + 1)
        roi = np.s_[y0:y1, x0:x1]
        if min(y1 - y0, x1 - x0) < 12:
            continue
        ref = cv2.cvtColor(warps[1][roi], cv2.COLOR_BGR2GRAY)
        side = cv2.cvtColor(warps[i][roi], cv2.COLOR_BGR2GRAY)
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ref, side, None)
        distance = cv2.distanceTransform((~seam[roi]).astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.clip(1. - distance / radius, 0., 1.)
        length = np.linalg.norm(flow, axis=2)
        flow *= (weight * np.minimum(1., 12. / np.maximum(length, 1e-9)))[..., None]
        yy, xx = np.indices(weight.shape, dtype=np.float32)
        mx, my = xx + flow[..., 0], yy + flow[..., 1]
        support = cv2.remap(masks[i][roi].astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
        use = (weight > 0) & overlap[roi] & support
        offset[roi][use] = flow[use]
        corrected = cv2.remap(warps[i][roi], mx, my, cv2.INTER_LINEAR)
        result[i] = warps[i].copy()
        result[i][roi][use] = corrected[use]
    return result, offsets


def _q2_compose(warps, owners):
    result = np.zeros_like(warps[0])
    for i, image in enumerate(warps):
        take = owners == i
        result[take] = image[take]
    y, x = np.nonzero(owners >= 0)
    return result[y.min():y.max() + 1, x.min():x.max() + 1]


def q2(img_paths):
    """SIFT + RANSAC, cylindrical projection, APAP, and seam-local DIS."""
    images = _read_images(img_paths, 3)
    pairs = _q2_sift_pairs(images)
    scale = .9 * images[1].shape[1]
    _, warps, masks, _ = _q2_cylinder_layers(images, pairs, scale)
    owners = _q2_owners(masks)
    corrected, _ = _q2_dis(warps, masks, owners)
    return _q2_compose(corrected, owners)
