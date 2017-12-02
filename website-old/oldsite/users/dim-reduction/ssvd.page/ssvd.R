
# standard SSVD
ssvd.svd <- function(x, k, p=25, qiter=0 ) { 

a <- as.matrix(x)
m <- nrow(a)
n <- ncol(a)
p <- min( min(m,n)-k,p)
r <- k+p

omega <- matrix ( rnorm(r*n), nrow=n, ncol=r)

y <- a %*% omega

q <- qr.Q(qr(y))

b<- t(q) %*% a

#power iterations
for ( i in 1:qiter ) { 
  y <- a %*% t(b)
  q <- qr.Q(qr(y))
  b <- t(q) %*% a
}

bbt <- b %*% t(b)

e <- eigen(bbt, symmetric=T)

res <- list()

res$svalues <- sqrt(e$values)[1:k]
uhat=e$vectors[1:k,1:k]

res$u <- (q %*% e$vectors)[,1:k]
res$v <- (t(b) %*% e$vectors %*% diag(1/e$values))[,1:k]

return(res)
}

#SSVD with Q=YR^-1 substitute.
# this is just a simulation, because it is suboptimal to verify the actual result
ssvd.svd1 <- function(x, k, p=25, qiter=0 ) { 

a <- as.matrix(x)
m <- nrow(a)
n <- ncol(a)
p <- min( min(m,n)-k,p)
r <- k+p

omega <- matrix ( rnorm(r*n), nrow=n, ncol=r)

# in reality we of course don't need to form and persist y
# but this is just verification
y <- a %*% omega

yty <- t(y) %*% y
R <- chol(yty, pivot = T)
q <- y %*% solve(R)

b<- t( q ) %*% a   

#power iterations
for ( i in 1:qiter ) { 
  y <- a %*% t(b)

  yty <- t(y) %*% y
  R <- chol(yty, pivot = T)
  q <- y %*% solve(R)
  b <- t(q) %*% a
}

bbt <- b %*% t(b)

e <- eigen(bbt, symmetric=T)

res <- list()

res$svalues <- sqrt(e$values)[1:k]
uhat=e$vectors[1:k,1:k]

res$u <- (q %*% e$vectors)[,1:k]
res$v <- (t(b) %*% e$vectors %*% diag(1/e$values))[,1:k]

return(res)
}


#############
## ssvd with pci options
ssvd.cpca <- function ( x, k, p=25, qiter=0, fixY=T ) { 

a <- as.matrix(x)
m <- nrow(a)
n <- ncol(a)
p <- min( min(m,n)-k,p)
r <- k+p


# compute median xi
xi<-colMeans(a)

omega <- matrix ( rnorm(r*n), nrow=n, ncol=r)

y <- a %*% omega

#fix y
if ( fixY ) { 
  #debug
  cat ("fixing Y...\n");

  s_o = t(omega) %*% cbind(xi)
  for (i in 1:r ) y[,i]<- y[,i]-s_o[i]
}


q <- qr.Q(qr(y))

b<- t(q) %*% a

# compute sum of q rows 
s_q <- cbind(colSums(q))

# compute B*xi
# of course in MR implementation 
# it will be collected as sums of ( B[,i] * xi[i] ) and reduced after.
s_b <- b %*% cbind(xi)


#power iterations
for ( i in 1:qiter ) { 

  # fix b 
  b <- b - s_q %*% rbind(xi) 

  y <- a %*% t(b)

  # fix y 
  if ( fixY )  
    for (i in 1:r ) y[,i]<- y[,i]-s_b[i]
  

  q <- qr.Q(qr(y))
  b <- t(q) %*% a

  # recompute s_{q}
  s_q <- cbind(colSums(q))

  #recompute s_{b}
  s_b <- b %*% cbind(xi)

}



#C is the outer product of S_q and S_b per doc
C <- s_q %*% t(s_b)

# fixing BB'
bbt <- b %*% t(b) -C -t(C) + sum(xi * xi)* (s_q %*% t(s_q))

e <- eigen(bbt, symmetric=T)

res <- list()

res$svalues <- sqrt(e$values)[1:k]
uhat=e$vectors[1:k,1:k]

res$u <- (q %*% e$vectors)[,1:k]

res$v <- (t(b- s_q %*% rbind(xi) ) %*% e$vectors %*% diag(1/e$values))[,1:k]

return(res)

}






