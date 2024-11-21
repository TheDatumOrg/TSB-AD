/********************************************
 * REVOLUTION 5.1.6 EXTENSION - PARALLAX
 * @version: 1.4 (10.03.2016)
 * @requires jquery.themepunch.revolution.js
 * @author ThemePunch
*********************************************/
(function($) {

var _R = jQuery.fn.revolution,
	_ISM = _R.is_mobile();

jQuery.extend(true,_R, {	
	/*callStaticDDDParallax: function(container,opt,li) {
		// STATIC 3D PARALLAX MOVEMENTS
	    if (opt.parallax && (opt.parallax.ddd_path=="static" || opt.parallax.ddd_path=="both")) {
			var coo = {},
				path = li.data('3dpath');
			coo.li = li;
			if (path.split(',').length>1) {
				coo.h = parseInt(path.split(',')[0],0);
				coo.v = parseInt(path.split(',')[1],0);				
				container.trigger('trigger3dpath',coo);
			}		
		}
	},*/

	checkForParallax : function(container,opt) {
		
		var _ = opt.parallax;

		if (_ISM && _.disable_onmobile=="on") return false;

		if (_.type=="3D" || _.type=="3d") {			
			punchgs.TweenLite.set(opt.c,{overflow:_.ddd_overflow});
			punchgs.TweenLite.set(opt.ul,{overflow:_.ddd_overflow});		
			if (opt.sliderType!="carousel" && _.ddd_shadow=="on") {
				opt.c.prepend('<div class="dddwrappershadow"></div>')
				punchgs.TweenLite.set(opt.c.find('.dddwrappershadow'),{force3D:"auto",transformPerspective:1600,transformOrigin:"50% 50%", width:"100%",height:"100%",position:"absolute",top:0,left:0,zIndex:0});			
			}
		}
		
		function setDDDInContainer(li) {
			if (_.type=="3D" || _.type=="3d") {
				li.find('.slotholder').wrapAll('<div class="dddwrapper" style="width:100%;height:100%;position:absolute;top:0px;left:0px;overflow:hidden"></div>');				
				li.find('.tp-parallax-wrap').wrapAll('<div class="dddwrapper-layer" style="width:100%;height:100%;position:absolute;top:0px;left:0px;z-index:5;overflow:'+_.ddd_layer_overflow+';"></div>');				

				// MOVE THE REMOVED 3D LAYERS OUT OF THE PARALLAX GROUP					
				li.find('.rs-parallaxlevel-tobggroup').closest('.tp-parallax-wrap').wrapAll('<div class="dddwrapper-layertobggroup" style="position:absolute;top:0px;left:0px;z-index:50;width:100%;height:100%"></div>');

				var dddw = li.find('.dddwrapper'),
					dddwl = li.find('.dddwrapper-layer'),
					dddwlbg = li.find('.dddwrapper-layertobggroup');

				

				dddwlbg.appendTo(dddw);
								
				if (opt.sliderType=="carousel") {
					 if (_.ddd_shadow=="on") dddw.addClass("dddwrappershadow");					 
					 punchgs.TweenLite.set(dddw,{borderRadius:opt.carousel.border_radius});
				}
				punchgs.TweenLite.set(li,{overflow:"visible",transformStyle:"preserve-3d",perspective:1600});
				punchgs.TweenLite.set(dddw,{force3D:"auto",transformOrigin:"50% 50%"});					
				punchgs.TweenLite.set(dddwl,{force3D:"auto",transformOrigin:"50% 50%",zIndex:5});					
				punchgs.TweenLite.set(opt.ul,{transformStyle:"preserve-3d",transformPerspective:1600});					
			}
		}

		opt.li.each(function() {
			setDDDInContainer(jQuery(this));						
		});

		if (_.type=="3D" || _.type=="3d" && opt.c.find('.tp-static-layers').length>0) {
			punchgs.TweenLite.set(opt.c.find('.tp-static-layers'),{top:0, left:0,width:"100%",height:"100%"});
			setDDDInContainer(opt.c.find('.tp-static-layers'));
		}

		for (var i = 1; i<=_.levels.length;i++)				
			opt.c.find('.rs-parallaxlevel-'+i).each(function() {					
				var pw = jQuery(this),
					tpw = pw.closest('.tp-parallax-wrap');												
				tpw.data('parallaxlevel',_.levels[i-1])
				tpw.addClass("tp-parallax-container");								
			});		

		
		if (_.type=="mouse" || _.type=="scroll+mouse" || _.type=="mouse+scroll" || _.type=="3D" || _.type=="3d") {
		
			container.mouseenter(function(event) {
				var currslide = container.find('.active-revslide'),
					t = container.offset().top,
					l = container.offset().left,
					ex = (event.pageX-l),
					ey =  (event.pageY-t);
				currslide.data("enterx",ex);
				currslide.data("entery",ey);
			});

			container.on('mousemove.hoverdir, mouseleave.hoverdir, trigger3dpath',function(event,data) {				
				var currslide = data && data.li ? data.li : container.find('.active-revslide');

				
				// CALCULATE DISTANCES
				if (_.origo=="enterpoint") {
					var	t = container.offset().top,
						l = container.offset().left;

					if (currslide.data("enterx")==undefined) currslide.data("enterx",(event.pageX-l));
					if (currslide.data("entery")==undefined) currslide.data("entery",(event.pageY-t));										

					var mh = currslide.data("enterx") || (event.pageX-l),
						mv = currslide.data("entery") || (event.pageY-t),
						diffh = (mh - (event.pageX - l)),
						diffv = (mv - (event.pageY - t)),
						s = _.speed/1000 || 0.4;
				} else {
					var	t = container.offset().top,
						l = container.offset().left,
						diffh = (opt.conw/2 - (event.pageX-l)),
						diffv = (opt.conh/2 - (event.pageY-t)),
						s = _.speed/1000 || 3;
				}
								
				/*if (event.type=="trigger3dpath") {
					diffh = data.h;
					diffv = data.v;					
					_.ddd_lasth = diffh;
					_.ddd_lastv = diffv;
				}*/

				if (event.type=="mouseleave") {
					diffh = _.ddd_lasth || 0;
					diffv = _.ddd_lastv || 0;
					s = 1.5;									
				}

				/*if (_.ddd_path=="static") {
					diffh = _.ddd_lasth || 0;
					diffv = _.ddd_lastv || 0;							
				}*/
				var pcnts = [];
				currslide.find(".tp-parallax-container").each(function(i){					
					pcnts.push(jQuery(this));
				});
				container.find('.tp-static-layers .tp-parallax-container').each(function(){
					pcnts.push(jQuery(this));
				});
				
				jQuery.each(pcnts, function() {
					var pc = jQuery(this),
						bl = parseInt(pc.data('parallaxlevel'),0),
						pl = _.type=="3D" || _.type=="3d" ? bl/200 : bl/100,
						offsh =	 diffh * pl,
						offsv =	 diffv * pl;		
						if (_.type=="scroll+mouse" || _.type=="mouse+scroll" ) 
							punchgs.TweenLite.to(pc,s,{force3D:"auto",x:offsh,ease:punchgs.Power3.easeOut,overwrite:"all"});
						else
							punchgs.TweenLite.to(pc,s,{force3D:"auto",x:offsh,y:offsv,ease:punchgs.Power3.easeOut,overwrite:"all"});
				});

				if (_.type=="3D" || _.type=="3d") {
					var sctor = '.tp-revslider-slidesli .dddwrapper, .dddwrappershadow, .tp-revslider-slidesli .dddwrapper-layer, .tp-static-layers .dddwrapper-layer';
					if (opt.sliderType==="carousel") sctor = ".tp-revslider-slidesli .dddwrapper, .tp-revslider-slidesli .dddwrapper-layer, .tp-static-layers .dddwrapper-layer";
					opt.c.find(sctor).each(function() {										
						var t = jQuery(this),
							pl = _.levels[_.levels.length-1]/200,										
							offsh =	diffh * pl,
							offsv =	diffv * pl,
							offrv = opt.conw == 0 ? 0 :  Math.round((diffh / opt.conw * pl)*100) || 0,
							offrh = opt.conh == 0 ? 0 : Math.round((diffv / opt.conh * pl)*100) || 0,										
							li = t.closest('li'),
							zz = 0,
							itslayer = false;

							if (t.hasClass("dddwrapper-layer")) {
								zz = _.ddd_z_correction || 65;
								itslayer = true;
							}

						if (t.hasClass("dddwrapper-layer")) {
							offsh=0;
							offsv=0;
						}
												
						if (li.hasClass("active-revslide") || opt.sliderType!="carousel")
							if (_.ddd_bgfreeze!="on" || (itslayer))								
								punchgs.TweenLite.to(t,s,{rotationX:offrh, rotationY:-offrv, x:offsh, z:zz,y:offsv,ease:punchgs.Power3.easeOut,overwrite:"all"});								  	
							else 								
								punchgs.TweenLite.to(t,0.5,{force3D:"auto",rotationY:0, rotationX:0, z:0,ease:punchgs.Power3.easeOut,overwrite:"all"});
						else 
							punchgs.TweenLite.to(t,0.5,{force3D:"auto",rotationY:0,z:0,x:0,y:0, rotationX:0, z:0,ease:punchgs.Power3.easeOut,overwrite:"all"});
																	
						if (event.type=="mouseleave")
						 	punchgs.TweenLite.to(jQuery(this),3.8,{z:0, ease:punchgs.Power3.easeOut});
					});
				}					
			});

			if (_ISM)
				window.ondeviceorientation = function(event) {
					var y = Math.round(event.beta  || 0)-70,
						x = Math.round(event.gamma || 0);

					var currslide = container.find('.active-revslide');

					if (jQuery(window).width() > jQuery(window).height()){
							var xx = x;
							x = y;
							y = xx;
					}

					var cw = container.width(),
						ch = container.height(),
						diffh = (360/cw * x),
				  		diffv = (180/ch * y),
				  		s = _.speed/1000 || 3,				  	
				  		pcnts = [];
					
					currslide.find(".tp-parallax-container").each(function(i){					
						pcnts.push(jQuery(this));
					});
					container.find('.tp-static-layers .tp-parallax-container').each(function(){
						pcnts.push(jQuery(this));
					});

				  	jQuery.each(pcnts, function() {
						var pc = jQuery(this),
							bl = parseInt(pc.data('parallaxlevel'),0),
							pl = bl/100,
							offsh =	 diffh * pl*2,
							offsv =	 diffv * pl*4;									
							punchgs.TweenLite.to(pc,s,{force3D:"auto",x:offsh,y:offsv,ease:punchgs.Power3.easeOut,overwrite:"all"});	
					});
					
					if (_.type=="3D" || _.type=="3d") {
						var sctor = '.tp-revslider-slidesli .dddwrapper, .dddwrappershadow, .tp-revslider-slidesli .dddwrapper-layer, .tp-static-layers .dddwrapper-layer';
						if (opt.sliderType==="carousel") sctor = ".tp-revslider-slidesli .dddwrapper, .tp-revslider-slidesli .dddwrapper-layer, .tp-static-layers .dddwrapper-layer";
						opt.c.find(sctor).each(function() {			
							var t = jQuery(this),
								pl = _.levels[_.levels.length-1]/200
								offsh =	diffh * pl,
								offsv =	diffv * pl*3,
								offrv = opt.conw == 0 ? 0 :  Math.round((diffh / opt.conw * pl)*500) || 0,
								offrh = opt.conh == 0 ? 0 : Math.round((diffv / opt.conh * pl)*700) || 0,
								li = t.closest('li'),
								zz = 0,
								itslayer = false;

							if (t.hasClass("dddwrapper-layer")) {
								zz = _.ddd_z_correction || 65;
								itslayer = true;
							}

							if (t.hasClass("dddwrapper-layer")) {
								offsh=0;
								offsv=0;
							}
												
							if (li.hasClass("active-revslide") || opt.sliderType!="carousel")
								if (_.ddd_bgfreeze!="on" || (itslayer))								
									punchgs.TweenLite.to(t,s,{rotationX:offrh, rotationY:-offrv, x:offsh, z:zz,y:offsv,ease:punchgs.Power3.easeOut,overwrite:"all"});								  	
								else 								
									punchgs.TweenLite.to(t,0.5,{force3D:"auto",rotationY:0, rotationX:0, z:0,ease:punchgs.Power3.easeOut,overwrite:"all"});
							else 
								punchgs.TweenLite.to(t,0.5,{force3D:"auto",rotationY:0,z:0,x:0,y:0, rotationX:0, z:0,ease:punchgs.Power3.easeOut,overwrite:"all"});
																	
							if (event.type=="mouseleave")
							 	punchgs.TweenLite.to(jQuery(this),3.8,{z:0, ease:punchgs.Power3.easeOut});
						});
					}
				}			 
		}
				
		_R.scrollTicker(opt,container);
		

	},
	
	scrollTicker : function(opt,container) {
		var faut;

		if (opt.scrollTicker!=true) {
			opt.scrollTicker = true;		
			if (_ISM) {		
				punchgs.TweenLite.ticker.fps(150);
				punchgs.TweenLite.ticker.addEventListener("tick",function() {_R.scrollHandling(opt);},container,false,1);
			} else {				
				jQuery(window).on('scroll mousewheel DOMMouseScroll', function() {				
					_R.scrollHandling(opt,true);					
				});
			}		
				
		}		
		_R.scrollHandling(opt, true);
	},



	//	-	SET POST OF SCROLL PARALLAX	-
	scrollHandling : function(opt,fromMouse) {	
		
		opt.lastwindowheight = opt.lastwindowheight || jQuery(window).height();

		var t = opt.c.offset().top,
			st = jQuery(window).scrollTop(),					
			b = new Object(),
			_v = opt.viewPort,
			_ = opt.parallax;

		
		if (opt.lastscrolltop==st && !opt.duringslidechange && !fromMouse) return false;
		//if (opt.lastscrolltop==st) return false;

		

		function saveLastScroll(opt,st) {			
			opt.lastscrolltop = st;			
		}
		punchgs.TweenLite.delayedCall(0.2,saveLastScroll,[opt,st]);

		b.top = (t-st);		
		b.h = opt.conh==0 ? opt.c.height() : opt.conh;		
		b.bottom = (t-st) + b.h;

		var proc = b.top<0 || b.h>opt.lastwindowheight ? b.top / b.h : b.bottom>opt.lastwindowheight ? (b.bottom-opt.lastwindowheight) / b.h : 0;
		opt.scrollproc = proc;

		if (_R.callBackHandling)
			_R.callBackHandling(opt,"parallax","start");

		

		if (_v.enable) {
			var area = 1-Math.abs(proc);
			area = area<0 ? 0 : area;
			// To Make sure it is not any more in %			
			if (!jQuery.isNumeric(_v.visible_area))
			 if (_v.visible_area.indexOf('%')!==-1) 
				_v.visible_area = parseInt(_v.visible_area)/100;
			

		 	if (1-_v.visible_area<=area) {
				if (!opt.inviewport) {
					opt.inviewport = true;
					_R.enterInViewPort(opt);
				}
			} else {
				if (opt.inviewport) {
					opt.inviewport = false;
					_R.leaveViewPort(opt);
				}
			}
		}

			
		// SCROLL BASED PARALLAX EFFECT 
		if (_ISM && opt.parallax.disable_onmobile=="on") return false;

		var pt = new punchgs.TimelineLite();
		pt.pause();

		if (_.type!="3d" && _.type!="3D") {
			if (_.type=="scroll" || _.type=="scroll+mouse" || _.type=="mouse+scroll") 
				opt.c.find(".tp-parallax-container").each(function(i) {
					var pc = jQuery(this),
						pl = parseInt(pc.data('parallaxlevel'),0)/100,
						offsv =	proc * -(pl*opt.conh) || 0;
					
					pc.data('parallaxoffset',offsv);					
					pt.add(punchgs.TweenLite.set(pc,{force3D:"auto",y:offsv}),0);
				});		

			opt.c.find('.tp-revslider-slidesli .slotholder, .tp-revslider-slidesli .rs-background-video-layer').each(function() {	
			
				var t = jQuery(this),
					l = t.data('bgparallax') || opt.parallax.bgparallax;				
					l = l == "on" ? 1 : l;						
					if (l!== undefined || l !== "off") {

						var pl = opt.parallax.levels[parseInt(l,0)-1]/100,
						offsv =	proc * -(pl*opt.conh) || 0;		


						if (jQuery.isNumeric(offsv))																					
							pt.add(punchgs.TweenLite.set(t,{position:"absolute",top:"0px",left:"0px",backfaceVisibility:"hidden",force3D:"true",y:offsv+"px"}),0);								
					}
			});
		}

		if (_R.callBackHandling)
			_R.callBackHandling(opt,"parallax","end");		
		
		pt.play(0);
	}
		
});



//// END OF PARALLAX EFFECT	
})(jQuery);